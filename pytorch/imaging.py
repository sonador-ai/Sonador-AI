import numpy as np
from typing import Optional, Union, List, Dict

import SimpleITK as sitk

from monai.config import DtypeLike, KeysCollection

from monai.data.image_reader import ImageReader, _copy_compatible_dict, _stack_images
from monai.transforms.transform import Transform, MapTransform
from monai.transforms.io.array import switch_endianness
from monai.utils import GridSampleMode, GridSamplePadMode, InterpolateMode, ensure_tuple, ensure_tuple_rep
from monai.transforms.utility.array import EnsureChannelFirst

from sonador.servers import SonadorImagingServer
from sonador.imaging.orthanc import ImagingSeries
from sonador3d.imaging import SonadorImagingVolume


DEFAULT_POST_FIX = 'meta_dict'
DEFAULT_SONADORAI_SERIES_POST = 'sonador_series'

DIRECTION_SHAPE_DEFAULT = {
	4: (2,2),
	9: (3,3),
	16: (4,4),
}


class SonadorImageReader(ImageReader):
	'''	Read imaging data from Sonador
	'''
	def __init__(self, *args,
			channel_dim: Optional[int]=None, reverse_indexing: bool=False,
			series_meta: bool=False, direction_shape=DIRECTION_SHAPE_DEFAULT, 
			cache_attr='sonadorai_volume', **kwargs):
		''' Initialize image reader instance.

			@input channel_dim (int, default=None): the channel dimension of the input image.
				Used to set original_channel_dim in the meta data, EnsureChallenFirstD reads this field.
				If None, `original_channel_dim` will be either `no_channel` or `-1`.
			@input reverse_indexing: toggles wheether to utilize a reversed spatial indexing convention for 
				the returned data array. If `False` the spatial indexing follows the numpy convention, otherwise 
				the spatial indexing is reversed to be compatible with ITK. Default is `False`.
				(This option does not affect metadata.)
			@input series_meta (bool, default=False): toggles whether to load the metadata of the DICOM series (using
				the meta from the first slice). Flag is checked only when loading DICOM series, default is `False`.
		'''
		super().__init__()
		self.args = args
		self.kwargs = kwargs
		self.channel_dim = channel_dim
		self.reverse_indexing = reverse_indexing
		self.series_meta = series_meta
		self.direction_shape = direction_shape
		self.cache_attr = cache_attr

	def verify_suffix(self, *args, **kwargs):
		raise NotImplementedError('Sonador imaging data is loaded in a raw format from Orthanc')

	def init_imagingvolume(self, *args,
			series: Optional[ImagingSeries]=None, volume: Optional[SonadorImagingVolume]=None, 
			cache=False, **kwargs):
		'''	Initialize Sonador imaging volume
		'''
		kwargs.update(self.kwargs)

		# Ensure that either or a volume or an imaging series was provided
		if series is None and volume is None:
			raise ValueError('Unable to read imaging data. Please provide a valid '
				+ 'imaging series or SonadorImagingVolume instance.')

		# Initialize volume instance. Order of preference:
		# 1. explicitly passed volume
		# 2. cached volume attribute
		# 3. initialize volume from series
		ivolume = volume or getattr(series, self.cache_attr, None) \
			or SonadorImagingVolume(series, *self.args, **kwargs)

		# Cache a copy of the volume to speed up downstream transform operations
		if series is None:
			series = ivolume.series
		if cache:
			setattr(series, self.cache_attr, ivolume)

		return ivolume

	def read(self, 
			series: Optional[ImagingSeries]=None, volume: Optional[SonadorImagingVolume]=None, cache=False):
		'''	Read image data from Sonador.  Either an imaging series or volume must be provided.
			(Volumes are used preferentially over the series if both are provided.)

			@input series (ImagingSeries, default=None): series for which the medical imaging data
				should be retrieved. (If a series is not provided, a volume is required.)
			@input volume (SonadorImagingVolume, default=None): volume for which the medical imaging
				data should be retrieved. (If a volume is not provided a series is required.)
			@input cache (bool, default=False): Toggles whether to cache a copy of the SonadorImagingVolume.
				If True, the volume initialized by the reader is added as an attribute to the series model 
				using the `cache_attr` property specifid at the time the reader was initialized.

			@returns SimpleITK.Image
		'''
		return self.init_imagingvolume(series=series, volume=volume, cache=cache).volume

	def get_data(self, img):
		'''	Extract data array and metadata from loaded image.

			@returns tuple
				- numpy.ndarray of image data
				- metadict: `affine`, `original_affine`, `spatial_shape`
		'''
		img_array: list[np.ndarray] = []
		compatible_meta: Dict = {}

		# Retrieve image data and headers
		data = self._get_array_data(img)
		img_array.append(data)
		
		# Retrieve metadata
		header = self._get_meta_dict(img)
		header['original_affine'] = self._get_affine(img)
		header['affine'] = header['original_affine'].copy()
		header['spatial_shape'] = self._get_spatial_shape(img)

		# Default to "no_channel" or -a
		if self.channel_dim is None:
			header['original_channel_dim'] = 'no_channel' if len(data.shape) == len(header['spatial_shape']) else -1
		else:
			header['original_channel_dim'] = self.channel_dim

		_copy_compatible_dict(header, compatible_meta)

		return _stack_images(img_array, compatible_meta), compatible_meta

	def _get_meta_dict(self, img, meta_dict=None) -> dict:
		'''	Get the metadata of the image and return as dict

			@input img (SimpleITK.Image): image object from which to retrieve metadata
		'''
		meta_dict = meta_dict or {}
		meta_dict.update({
			'spacing': np.asarray(img.GetSpacing())
		})
		img_array: List

		return meta_dict

	def _get_affine(self, img, lps_to_ras: bool=True):
		'''	Get or construct the affine matrix of the image, which can be used to correct spacing, orientation,
			or to execute spatial transforms.

			@input img (SimpleITK.Image): image object from which to load the data
		'''
		d = img.GetDirection()
		if not len(d) in self.direction_shape:
			raise ValueError('Unable to retrieve affine transform, invalid direction matrix')

		# Retrieve direction, spacing, and origin
		direction = np.reshape(d, self.direction_shape.get(len(d)))
		spacing = np.asarray(img.GetSpacing())
		origin = np.asarray(img.GetOrigin())

		sr = min(max(direction.shape[0], 1), 3)
		affine: np.ndarray = np.eye(sr+1)
		affine[:sr,:sr] = direction[:sr,:sr] @ np.diag(spacing[:sr])
		affine[:sr, -1] = origin[:sr]

		# ITK to nibabel affine
		flip_diag = [[-1, -1], [-1, -1, 1], [-1, -1, 1, 1]][sr-1]
		affine = np.diag(flip_diag) @ affine
		return affine

	def _get_spatial_shape(self, img):
		'''	Get the spatial shape of img

			@input img (SimpleITK.Image): image object for which to retrieve the spatial shape
		'''	
		d = img.GetDirection()
		if not len(d) in self.direction_shape:
			raise ValueError('Unable to retrieve spatial shape of img, invalid direction matrix')

		sr = np.reshape(d, self.direction_shape.get(len(d))).shape[0]
		sr = max(min(sr, 3), 1)
		_size = list(img.GetSize())

		# Remove channel dim (if present)
		if self.channel_dim is not None:
			_size.pop(self.channel_dim)

		return np.asarray(_size[:sr])

	def _get_array_data(self, img):
		'''	Get the raw array data of the image, converted to numpy. Following PyTorch conventions, the
			returned aray data has contiguous channels. This means that for RGB images, all red channel image pixels
			are contiguous in memory. The last axis of the returned array is the channel axis.

			@input img (SimpleITK.Image): image object from which to convert the data
		'''
		img_array = sitk.GetArrayFromImage(img)

		# Spatial images
		if img.GetNumberOfComponentsPerPixel() == 1:
			return img_array if self.reverse_indexing else img_array.T

		# Handle multi-channel images
		return img_array if self.reverse_indexing else np.moveaxis(img_array.T, 0, -1)


class LoadSonadorImage(Transform):
	'''	Load imaging series from Sonador
	'''
	def __init__(self, 
			image_only: bool=False, dtype: DtypeLike=np.float32, ensure_channel_first: bool=False, *args, **kwargs):
		self.image_only = image_only
		self.dtype = dtype
		self.ensure_channel_first = ensure_channel_first

		# Initialize reader instance
		self.reader = self.init_reader(*args, **kwargs)
		return

	def init_reader(self, *args, **kwargs):
		return SonadorImageReader(*args, **kwargs)

	def __call__(self, series: Optional[ImagingSeries]=None, volume: Optional[SonadorImagingVolume]=None,
			reader: Optional[SonadorImageReader]=None, **kwargs):
		'''	Load image file and metadata from the provided volume or series.
		'''
		reader = reader or self.reader
		img = reader.read(series=series, volume=volume, **kwargs)

		img_array, meta = reader.get_data(img)
		img_array = img_array.astype(self.dtype, copy=False)

		if not isinstance(meta, dict):
			raise ValueError('metadata must be a dictionary')

		# Make sure all elements in the metadata are little endian
		meta = switch_endianness(meta, '<')
		if self.ensure_channel_first:
			img_array = EnsureChannelFirst()(img_array, meta)

		if self.image_only:
			return img_array

		return img_array, meta


class LoadSonadorImaged(MapTransform):
	''' Load imaging series data from Sonador using MONAI dictionary format.
	'''
	def __init__(self,
			imaging_server, *args,
			keys: KeysCollection, 
			dtype: DtypeLike=np.float32,
			meta_keys: Optional[KeysCollection] = None,
			meta_key_postfix: str=DEFAULT_POST_FIX,
			overwriting: bool=False,
			image_only: bool=False, 
			ensure_channel_first: bool=False,
			allow_missing_keys: bool=False, 
			imaging_server_cache: bool=False, **kwargs) -> None:
		'''	Dictionary-based data loader for working with imaging data from Sonador.

			The transform returns an image data array if `image_only` is True, or a tuple of two elements
			containing the data array and the metadata in a dictionary format otherwise.
			
			@input imaging_server (sonador.servers.SonadorImagingServer): image server from which
				to retrieve data
			@input keys: keys of the items to be transformed. Values associated with the key should
				reference an Orthanc series UID (`resource.pk`) available from imaging server.
			@input image_only (bool, default=False): If True, returns only the image volume, otherwise
				return image data array and header dict.
			@input ensure_channel_first (bool, default=False): if `True` and loaded both image array and metadata, 
				automatically convert the image array shape to `channel first`.
			@input imaging_server_cache (bool, default=False): If `True`, the Sonador imaging server cache
				will be used to retrieve resources and resource data.
		'''
		super().__init__(keys, allow_missing_keys)
		self._loader = self.init_loader(
			*args, image_only=image_only, dytpe=dtype, ensure_channel_first=ensure_channel_first, **kwargs)

		# Sonador imaging server: used to medical imaging data
		self.imaging_server = imaging_server

		if not isinstance(meta_key_postfix, str):
			raise TypeError('meta_key_postfix must be a str, invalid type: %s' % type(meta_key_postfix))
		self.meta_keys = ensure_tuple_rep(None, len(self.keys)) if meta_keys is None else ensure_tuple(meta_keys)
		if len(self.keys) != len(self.meta_keys):
			raise ValueError('meta_keys muyst have the same length as keys')
		self.meta_key_postfix = ensure_tuple_rep(meta_key_postfix, len(self.keys))

		self.image_only = image_only
		self.dtype = dtype
		self.ensure_channel_first = ensure_channel_first
		self.overwriting = overwriting
		self.imaging_server_cache = imaging_server_cache

	def init_loader(self, *args, **kwargs):
		'''	Initialize loader instance
		'''
		return LoadSonadorImage(*args, **kwargs)

	def get_sonador_series(self, key, d, *args, **kwargs) -> ImagingSeries:
		'''	Retrieve Sonador series reference

			@input key: data dictionary key for which the series needs to be retrieved
			@input d (dict): MONAI data dictionary

			@returns sonador.imaging.orthanc.ImagingSeries
		'''
		# Retrieve imaging series
		if isinstance(d[key], ImagingSeries):
			iseries = d[key]

		# Retrieve series using resource UID
		elif isinstance(d[key], str):
			iseries = self.imaging_server.get_series(d[key], cache=self.imaging_server_cache)

		# Unsupported type
		else:
			raise TypeError('Unsupported type for data dict key "%s"' % key)

		return iseries

	def __call__(self, data, reader: Optional[SonadorImageReader]=None, **kwargs):
		'''	Retrieve image data
		'''
		d = dict(data)

		# Retrieve imaging data from server for specified keys
		for key, meta_key, meta_key_postfix in self.key_iterator(d, self.meta_keys, self.meta_key_postfix):

			# Retrieve imaging series
			iseries = self.get_sonador_series(key, d)

			# Retrieve pixel data
			data = self._loader(
				series=iseries, reader=reader, cache=self.imaging_server_cache, **kwargs)

			# Image data without metadata
			if self._loader.image_only:

				# Ensure correct data type
				if not isinstance(data, np.ndarray):
					raise ValueError('Loader must return a numpy.ndarray when image_only=True')

				# Write pixel data to dict
				d[key] = data

			# Pixel and metadata
			else:

				# Ensure correct data type
				if not isinstance(data, (tuple, list)):
					raise ValueError('Loader must return tuple or list when image_only=False')

				# Write pixel data to dict
				d[key] = data[0]
				if not isinstance(data[1], dict):
					raise ValueError("metadata must be returned as a dict")

				# Wirte meta to dict
				meta_key = meta_key or f'{key}_{meta_key_postfix}'
				if meta_key in d and not self.overwriting:
					raise KeyError(f'Metadata with key {meta_key} already exists and overwriting is disabled')

				d[meta_key] = data[1]

		return d
