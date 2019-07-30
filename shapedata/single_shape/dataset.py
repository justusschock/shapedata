# author: Justus Schock (justus.schock@rwth-aachen.de)

from delira.data_loading import AbstractDataset
from collections import OrderedDict
import os
import random
import numpy as np
from .data_processing import SingleImage2D as SingleImage
from multiprocessing import Pool
from functools import partial
from skimage.transform import AffineTransform
from ..utils import make_dataset, is_image_file, LMK_EXTENSIONS, IMG_EXTENSIONS_2D


def default_loader(data: str, img_size: tuple, crop=None, extension=None,
                   rotate=None, cached=False, random_offset=False,
                   random_scale=False, point_indices=None):
    """
    Helper Function to load single sample

    Parameters
    ----------
    data : str or :class:`SingleImage2D`
        image file to load
    img_size : tuple
        image size for resizing
    crop : None or float
        if None: nor cropping will be applied
        if float: specifies boundary proportion for cropping
    extension : str or None:
        specifiying the extension
    rotate : int or None
        specifies to image rotation (in degrees)
    cached : bool
        whether or not the data is already cached
    random_offset : bool or float
        if bool: must be False -> No Random Shift is applied
        if float: specifies the maximal number of pixels to shift
    random_scale : bool or float
        if bool: must be False -> No random scaling is applied
        if float: specifies the maximum amount of scaling
    point_indices : None or Iterable
        if None: All landmarks are returned
        if Iterable: only landmarks corresponding to indices are returned

    Returns
    -------
    np.ndarray
        image
    np.ndarray
        landmarks

    """
    if not cached:
        _data = SingleImage.from_files(data, extension=extension)
    else:
        _data = data
    return preprocessing(_data, img_size, crop, rotate, random_offset,
                         random_scale, point_indices)


def preprocessing(img: SingleImage, img_size: tuple, crop=None, rotate=None,
                  random_offset=False, random_scale=False, point_indices=None):
    """
    Helper Function to preprocess a single sample

    Parameters
    ----------
    img ::class:`SingleImage2D`
        image file to preprocess
    img_size : tuple
        image size for resizing
    crop : None or float
        if None: nor cropping will be applied
        if float: specifies boundary proportion for cropping
    extension : str or None:
        specifiying the extension
    rotate : int or None
        specifies to image rotation (in degrees)
    random_offset : bool or float
        if bool: must be False -> No Random Shift is applied
        if float: specifies the maximal number of pixels to shift
    random_scale : bool or float
        if bool: must be False -> No random scaling is applied
        if float: specifies the maximum amount of scaling
    point_indices : None or Iterable
        if None: All landmarks are returned
        if Iterable: only landmarks corresponding to indices are returned

    Returns
    -------
    np.ndarray
        image
    np.ndarray
        landmarks
        
    """
    _data = img
    if crop is not None or rotate or random_scale or random_offset:
        if crop is None:
            crop = 0

        if rotate:
            _data = _data.crop_to_landmarks(2)
            _data = _data.rotate(rotate)

        if random_scale:
            scale = [random.uniform(1 - random_scale, 1 + random_scale)
                     for i in range(len(_data.img.shape[:-1]))]
        else:
            scale = [1] * len(_data.img.shape[:-1])

        if random_offset:
            offset = [random.randint(-random_offset, random_offset)
                      for i in range(len(_data.img.shape[:-1]))]
        else:
            offset = 0

        affine_trafo = AffineTransform(scale=scale, translation=offset)

        bounds = _data.get_landmark_bounds(_data.lmk)

        _bound_pts = np.array([[bounds[1], bounds[0]],
                               [bounds[3], bounds[2]]])

        transformed_pts = affine_trafo(_bound_pts)

        min_y, min_x = transformed_pts[:, 1].min(), transformed_pts[:, 0].min()
        max_y, max_x = transformed_pts[:, 1].max(), transformed_pts[:, 0].max()

        _data_bounds = [min_y, min_x, max_y, max_x]

        # constrain bounds to image bounds
        _data_bounds = [min(max(0, _bound), min(_data.img.shape[:-1]))
                        for _bound in _data_bounds]

        range_y, range_x = _data_bounds[2] - _data_bounds[0], \
                           _data_bounds[3] - _data_bounds[1]

        total_range = max(range_x, range_y) * (1 + crop)

        _data_bounds = [
            _data_bounds[0] + range_y / 2 - total_range / 2,
            _data_bounds[1] + range_x / 2 - total_range / 2,
            _data_bounds[0] + range_y / 2 + total_range / 2,
            _data_bounds[1] + range_x / 2 + total_range / 2,
            ]

        _data = _data.crop(*_data_bounds)

    _data = _data.resize(img_size)
    _data = _data.to_grayscale()

    if point_indices:
        lmk = _data.lmk[point_indices]
    else:
        lmk = _data.lmk
    return _data.img.transpose(2, 0, 1), lmk


class ShapeDataset(AbstractDataset):
    """
    Dataset to load image and corresponding shape
    """

    def __init__(self, data_path, img_size, crop=None,
                 extension=None,
                 rotate=None, cached=False, random_offset=False,
                 random_scale=False,
                 point_indices=None):
        """

        Parameters
        ----------
        data_path : str
            path to shapedata directory
        img_size : tuple
            image size
        crop : float or None
            if None: nor cropping will be applied
            if float: specifies boundary proportion for cropping
        extension : str or None
            specifies the landmark extension
        rotate : int or None
            specifies to image rotation (in degrees)
        cached : bool
            whether or not the data is already cached
        random_offset : bool or float
            if bool: must be False -> No Random Shift is applied
            if float: specifies the maximal number of pixels to shift
        random_scale : bool or float
            if bool: must be False -> No random scaling is applied
            if float: specifies the maximum amount of scaling
        point_indices : None or Iterable
            if None: All landmarks are returned
            if Iterable: only landmarks corresponding to indices are returned

        """

        super().__init__(data_path, default_loader)

        self.data_path = data_path
        self.img_size = (img_size, img_size) if isinstance(img_size, int) \
            else img_size
        self.crop = crop
        self.point_indices = point_indices
        self.extension = extension
        self.rotate = rotate
        self.random_offset = random_offset
        self.random_scale = random_scale

        if rotate:
            random.seed(1234)

        self.cached = cached

        img_files = make_dataset(data_path)

        if self.cached:
            with Pool() as p:
                data = list(p.map(partial(SingleImage.from_files,
                                          extension=extension),
                                  img_files))

        else:
            data = img_files

        self.data = data

    def _make_dataset(self, path):
        return tuple([os.path.join(path, x) for x in os.listdir(path)
                      # check if file exists
                      if (os.path.isfile(os.path.join(path, x)) and
                          # check if landmark file exists
                          any([os.path.isfile(os.path.join(
                              path, os.path.splitext(x) + ext))
                              for ext in LMK_EXTENSIONS])
                          # check if file is image file
                          and any([x.endswith(ext)
                                   for ext in IMG_EXTENSIONS_2D]))])

    def __getitem__(self, index):
        """
        Returns a dict containing a single sample
        
        Parameters
        ----------
        index : int
            index specifying the sample to return
        
        Returns
        -------
        dict
            dictionary containing the image under the key 'data' and the 
            landmarks under the key 'label'
            
        """

        if self.rotate:
            rot_angle = random.randint(-self.rotate, self.rotate)
        else:
            rot_angle = None
        # _img, _label = default_loader(self.img_files[index], self.img_size)
        _img, _label = self._load_fn(self.data[index], self.img_size,
                                     self.crop, self.extension, rot_angle,
                                     self.cached,
                                     self.random_offset,
                                     self.random_scale,
                                     self.point_indices)

        return {"data": _img, "label": _label}

    def __len__(self):
        return len(self.data)
