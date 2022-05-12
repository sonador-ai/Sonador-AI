from typing import Optional, Union, List, Dict
import SimpleITK as sitk

from client.utils.general import first

from sonador.imaging.orthanc import ImagingSeries
from sonador3d.imaging import SonadorImagingVolume

from .imaging import SonadorImageReader, LoadSonadorImage, LoadSonadorImaged


class SonadorSegmentationReader(SonadorImageReader):
	'''	Read segmentation data from Sonador
	'''
	def __init__(self, *args, segment_labels=None, **kwargs):
		super().__init__(*args, **kwargs)
		self.segment_labels = segment_labels

		if self.segment_labels is not None and not isinstance(self.segment_labels, (str, list, tuple)):
			raise TypeError('Unable to initialize transform instance, invalid segment label type. '
				+ 'Segment labels must be a string or an iterable of strings.')

	def read(self, *args, series: Optional[ImagingSeries]=None, volume: Optional[SonadorImagingVolume]=None,
			cache=False, **kwargs):
		'''	Read segmentation data from Sonador. Either an imaging series or volume must be provided.
			(Volumes are used preferentially over the series if both are provided.)

			@input series (ImagingSeries, default=None): series for which the medical imaging data
				should be retrieved. (If a series is not provided, a volume is required.)
			@input volume (SonadorImagingVolume, default=None): volume for which the medical imaging
				data should be retrieved. (If a volume is not provided a series is required.)
			@input cache (bool, default=False): Toggles whether to cache a copy of the SonadorImagingVolume.
				If True, the volume initialized by the reader is added as an attribute to the series
				model using the `cache_attr` property specified at the time the reader was initialized.

			@returns SimpleITK.Image
		'''
		# Initialize imaging volume and retrieve labelmap image
		ivolume = self.init_imagingvolume(
			*args, series=series, volume=volume, cache=cache, **kwargs)

		# Retrieve all segmentations for the imaging series
		if self.segment_labels is None:
			labelmap_image = ivolume.labelmaps_image

		else:

			# Retrieve single labelmap
			if isinstance(self.segment_labels, str):
				lmeta = first(
					ivolume.labelmaps.keys(), key=lambda m: (m.label or '') == self.segment_labels)
				labelmap_image = ivolume.labelmaps.get(lmeta)

			# Retrieve/combine multiple labelmaps: labelmap values correspond to the position
			# provided in `segment_labels` iterable.
			elif isinstance(self.segment_labels, (list, tuple)):

				# Match meta labels and pack
				labelmap_image = ivolume._init_empty_image()
				for i, lmeta in enumerate([m for m in ivolume.labelmaps.keys() if m.label in self.segment_labels]):
					labelmap_image += (i+1)*sit.Cast(ivolume.labelmaps.get(lmeta), labelmap_image.GetPixelIDValue())

		# Create placeholder labelmap if unable to retrieve segments
		if labelmap_image is None:
			labelmap_image = ivolume._init_empty_image()

		return labelmap_image


class LoadSonadorSegmentation(LoadSonadorImage):
	'''	Load segmentation data from Sonador associated with an imaging series
	'''
	def init_reader(self, *args, **kwargs):
		return SonadorSegmentationReader(*args, **kwargs)


class LoadSonadorSegmentationd(LoadSonadorImaged):
	'''	Load imaging series segmentation data from Sonador using MONAI dictionary format.
	'''
	def init_loader(self, *args, **kwargs):
		return LoadSonadorSegmentation(*args, **kwargs)
