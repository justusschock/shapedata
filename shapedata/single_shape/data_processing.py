# author: Justus Schock (justus.schock@rwth-aachen.de)

import os
from sklearn.decomposition import PCA
from skimage import io as sio
from skimage.transform import AffineTransform, warp_coords
import numpy as np
import glob
from matplotlib import pyplot as plt
from multiprocessing import Pool
import SimpleITK as sitk
from delira.utils.imageops import sitk_resample_to_spacing

from ..utils import is_landmark_file, is_image_file, LMK_EXTENSIONS, \
    IMG_EXTENSIONS_2D
from tqdm import tqdm
from ..io import ljson_importer, ljson_exporter, pts_importer, pts_exporter
from ..base_data_processing import BaseSingleImage, AbstractSingleImage
from copy import deepcopy


class SingleImage2D(BaseSingleImage):
    """
    Holds Single Image

    """

    def __init__(self, img, lmk=None, **kwargs):
        """

        Parameters
        ----------
        img : np.ndarray
            actual image pixels
        lmk : np.ndarray
            landmarks
        kwargs :
            additional kwargs like file paths

        """
        super().__init__(img)

        self.lmk = lmk

        for key, val in kwargs.items():
            setattr(self, key, val)

    @classmethod
    def from_npy_files(cls, file, **kwargs):
        """
        Create class from image file
        Parameters
        ----------
        file : str
            path to image file
        **kwargs :
            additional keyword arguments

        Returns
        -------
        :class:`SingleImage2D`
            class instance 

        """

        img, lmk = None, None

        sitk_image = sitk.ReadImage(file)
        if kwargs.get("resample_spacing", False):
            sitk_image = sitk_resample_to_spacing(
                sitk_image, [1.0]*sitk_image.GetDimension())

        img = sitk.GetArrayFromImage(sitk_image)
        img_file = file
        lmk_file = None
        for ext in LMK_EXTENSIONS:
            curr_ext = "." + file.rsplit(".", maxsplit=1)[-1]

            _lmk_file = file.replace(curr_ext, ext)
            if os.path.isfile(_lmk_file):
                lmk = np.loadtxt(_lmk_file)
                lmk_file = _lmk_file

        return cls(img, lmk, img_file=img_file, lmk_file=lmk_file, **kwargs)

    @classmethod
    def from_ljson_files(cls, img_file, **kwargs):
        """
        Creates class from menpo pts landmarks and image

        Parameters
        ----------
        img_file : str
            image file to load
        **kwargs :
            additional keyword arguments

        Returns
        -------
        :class:`SingleImage2D`
            class instance

        """
        sitk_image = sitk.ReadImage(img_file)
        if kwargs.get("resample_spacing", False):
            sitk_image = sitk_resample_to_spacing(
                sitk_image, [1.0]*sitk_image.GetDimension())

        img = sitk.GetArrayFromImage(sitk_image)

        pt_file = img_file.rsplit(".", 1)[0] + ".ljson"
        if os.path.isfile(pt_file):
            points = ljson_importer(pt_file)
        else:
            points = None
        return cls(img, points, img_file=img_file, lmk_file=pt_file, **kwargs)

    @classmethod
    def from_pts_files(cls, img_file, **kwargs):
        """
        Creates class from menpo ljson landmarks and image

        Parameters
        ----------
        img_file : str
            image file to load
        **kwargs :
            additional keyword arguments

        Returns
        -------
        :class:`SingleImage2D`
            class instance

        """
        sitk_image = sitk.ReadImage(img_file)
        if kwargs.get("resample_spacing", False):
            sitk_image = sitk_resample_to_spacing(
                sitk_image, [1.0]*sitk_image.GetDimension())

        img = sitk.GetArrayFromImage(sitk_image)
        pt_file = img_file.rsplit(".", 1)[0] + ".pts"
        if os.path.isfile(pt_file):
            points = pts_importer(pt_file)
        else:
            points = None
        return cls(img, points, img_file=img_file, lmk_file=pt_file, **kwargs)

    def _save_landmarks(self, filepath, lmk_type, **kwargs):
        """
        Saves landmarks to file

        Parameters
        ----------
        filepath : str
            path to file the landmarks should be saved to
        lmk_type : str
            specifies the type of landmark file
        **kwargs
            additional keyword arguments passed to save function

        Raises
        ------
        ValueError
            no valid landmarktype is given

        """
        if lmk_type.lower() == 'ljson':
            if not filepath.endswith('.ljson'):
                filepath = filepath + ".ljson"

            return ljson_exporter(self.lmk, filepath, **kwargs)

        elif lmk_type.lower() == 'pts':
            if not filepath.endswith('.pts'):
                filepath = filepath + '.pts'

            return pts_exporter(self.lmk, filepath, **kwargs)

        elif lmk_type.lower() == 'npy':
            if not filepath.endswith('.txt'):
                filepath = filepath + ".txt"

            return np.savetxt(filepath, self.lmk, **kwargs)

        else:
            raise ValueError("Landmarktype not supported!")

    def _transform_lmk(self, transformation: AffineTransform):
        """
        Apply transformation inplace to landmarks

        Parameters
        ----------
        transformation : :class:`skimage.transform.AffineTransform`
            transformation to apply

        Returns
        -------
        :class:`BaseSingleImage`
            Image with Transformed Landmarks

        """
        if self.lmk is not None:
            # flip coords for transformation and flip back afterwards
            self.lmk = transformation(np.ascontiguousarray(self.lmk[:, [1, 0]])
                                      )[:, [1, 0]]

        return self.cartesian_coordinates()

    def homogeneous_coordinates(self):
        """
        Transforms landmark coordinates inplace to homogeneous coordinates

        Returns
        -------
        :class:`SingleImage2D`
            Image with Landmarks in Homogeneous Coordinates

        """
        if self.is_cartesian:
            self.lmk = np.hstack([self.lmk, np.ones((self.lmk.shape[0], 1))])
            self.is_homogeneous = True

        return self

    def cartesian_coordinates(self):
        """
        Transforms landmark coordinates inplace to cartesian coordinates

        Returns
        -------
        class:`SingleImage2D`
            Image with Landmarks in cartesian Coordinates

        """
        if self.is_homogeneous:
            self.lmk = self.lmk[:, :-1] / self.lmk[:, -1].reshape(
                self.lmk.shape[0], 1)
            self.is_cartesian = True
        return self

    def normalize_rotation(self, index_left, index_right, **kwargs):
        """
        normalizes rotation based on two keypoints

        index_left : int
            landmark-index of the left point
        index_right : int
            landmark-index of the right point
        **kwargs:
            additional keyword arguments (passed to :meth:`warp`)

        Returns
        -------
        :class:`SingleImage2D`
            normalized image

        """
        return self._normalize_rotation(self.lmk, index_left, index_right,
                                        **kwargs)

    def view(self, view_landmarks=False, create_fig=False, **kwargs):
        """
        Shows image (and optional the landmarks)

        Parameters
        ----------
        view_landmarks : bool
            whether or not to show the landmarks
        **kwargs :
            additional keyword arguments (are passed to imshow)

        Returns
        -------
        :class:`Figure`
            figure with plot

        """
        if create_fig:
            fig = plt.figure()
            ax = fig.gca()
        else:
            fig = plt.gcf()
            ax = plt.gca()

        ax.imshow(self.img.squeeze(), **kwargs)

        if view_landmarks and self.lmk is not None:
            marker_size = min(max(self.img.shape)//100, 15)
            ax.scatter(self.lmk[:, 1], self.lmk[:, 0], c="C0", s=marker_size)

        return fig

    def _crop(self, min_y, min_x, max_y, max_x):
        """
        Implements actual cropping inplace

        Parameters
        ----------
        min_y : int
            minimum y value
        min_x : int
            minimum x value
        max_y : int
            maximum y value
        max_x : int
            maximum x value

        Returns
        -------
        :class:`SingleImage2D`
            cropped image

        """
        self.img = self.img[int(min_y): int(max_y), int(min_x): int(max_x)]

        if self.lmk is not None:
            self.lmk = self._crop_lmks(self.lmk, int(min_y), int(min_x),
                                       int(max_y), int(max_x))

        return self

    def _crop_to_landmarks(self, proportion=0., **kwargs):
        """
        Crop to landmarks

        Parameters
        ----------
        proportion : float
            Cropping Proportion
        **kwargs :
            additional keyword arguments (ignored here)

        Returns
        -------
        :class:`SingleImage2D`
            Cropped Image

        """
        min_y, min_x, max_y, max_x = self.get_landmark_bounds(self.lmk)

        range_x = max_x - min_x
        range_y = max_y - min_y

        max_range = max(range_x, range_y) * (1 + proportion)

        center_x = min_x + range_x / 2
        center_y = min_y + range_y / 2

        return self.crop(center_y - max_range / 2,
                         center_x - max_range / 2,
                         center_y + max_range / 2,
                         center_x + max_range / 2)


class DataProcessing(object):
    """
    Process multiple SingleImages

    See Also
    --------
    :class:`SingleImage2D`

    """

    def __init__(self, samples, dim=2, **kwargs):
        """

        Parameters
        ----------
        samples : list
            list of SingleImages
        dim : int
            number of image dimensions
        **kwargs :
            additional keyword arguments

        """
        super().__init__()
        self.samples = samples
        self.dim = dim
        for key, val in kwargs.items():
            setattr(self, key, val)

    @classmethod
    def from_dir(cls, data_dir, verbose=True, n_jobs=None, n_dim=2):
        """
        create class instance from directory

        Parameters
        ----------
        data_dir : str
            directory where shapedata is stored
        verbose : bool
            whether or not to print current progress
        n_jobs : int
            number of jobs for loading data (default: None -> all available 
            CPUs are used)
        n_dim : int
            Integer indicating the dimensionality of the image (default: 2)

        Returns
        -------
        :class:`DataProcessing`
            class instance

        """

        if verbose:
            print("Loading shapedata from %s" % data_dir)
            wrapper_fn = tqdm
        else:
            def linear_wrapper(x):
                return x

            wrapper_fn = linear_wrapper

        files = cls._get_files(data_dir, IMG_EXTENSIONS_2D)
        if n_jobs == 1:
            samples = []
            for file in wrapper_fn(files):
                samples.append(SingleImage2D.from_files(file))

        else:
            with Pool(n_jobs) as p:
                samples = p.map(SingleImage2D.from_files, files)

        return cls(samples=samples, dim=n_dim)

    @property
    def landmarks(self):
        """
        get list of samples' landmarks

        Returns
        -------
        list
            landmarks

        """
        return [tmp.lmk for tmp in self.samples]

    @property
    def images(self):
        """
        get list of samples' pixels

        Returns
        -------
        list
            pixels

        """
        return [tmp.img for tmp in self.samples]

    def resize(self, img_size):
        """
        resize all samples

        Parameters
        ----------
        img_size : tuple
            new image size

        """
        for idx, sample in enumerate(self.samples):
            self.samples[idx] = sample.resize(img_size)

    @staticmethod
    def _get_files(directory, extensions):
        """
        return files with extensions

        Parameters
        ----------
        directory : str
            directory containing the files
        extensions : list
            list of strings specifying valid extensions

        Returns
        -------
        list
            valid files

        """

        files = []

        if not isinstance(extensions, list):
            extensions = [extensions]
        for ext in extensions:
            ext = ext.strip(".")
            files += glob.glob(directory + "/*." + ext)
        files.sort()
        return files

    def __getitem__(self, index):
        return self.samples[index]

    def __setitem__(self, index, value):
        self.samples[index] = value

    def __len__(self):
        return len(self.samples)

    def lmk_pca(self, scale: bool, center: bool, pt_indices=[], *args,
                **kwargs):
        """
        perform PCA on samples' landmarks

        Parameters
        ----------
        scale : bool
            whether or not to scale the principal components with the
            corresponding eigen value
        center : bool
            whether or not to substract mean before pca
        pt_indices : int
            indices to include into PCA (if empty: include all points)
        args : list
            additional positional arguments (passed to pca)
        **kwargs :
            additional keyword arguments (passed to pca)

        Returns
        -------
        np.array
            eigen_shapes

        """

        landmarks = np.asarray(self.landmarks).copy()

        if pt_indices:
            landmarks = landmarks[:, pt_indices, :]

        if center:
            mean = np.mean(landmarks.reshape(-1, landmarks.shape[-1]), axis=0)
            landmarks = landmarks - mean
        landmarks_transposed = landmarks.transpose((0, 2, 1))

        reshaped = landmarks_transposed.reshape(landmarks.shape[0], -1)
        pca = PCA(*args, **kwargs)
        pca.fit(reshaped)

        if scale:
            components = pca.components_ * pca.singular_values_.reshape(-1, 1)
        else:
            components = pca.components_

        return np.array([pca.mean_] + list(components)).reshape(
            components.shape[0] + 1,
            *landmarks_transposed.shape[1:]).transpose(0, 2, 1)
