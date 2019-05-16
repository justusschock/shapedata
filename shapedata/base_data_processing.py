# author: Justus Schock (justus.schock@rwth-aachen.de)

import os
import numpy as np
from abc import abstractmethod
from copy import deepcopy

from skimage.color import rgb2gray
from skimage.transform import AffineTransform, warp
from skimage.io import imsave

# TODO: Add support for connectivity-Information


class AbstractSingleImage(object):
    """
    Abstract Class to define a SingleImage-API
    
    """

    def __init__(self):
        self._img = None
        self._transformation_history = []
        self.is_cartesian = True

    @property
    def img(self):
        """
        Property to get the actual image pixels
        
        Returns
        -------
        np.array
            image pixels

        """

        return self._img

    @img.setter
    @abstractmethod
    def img(self, new_img):
        """
        Setter for the ``img`` property
        
        Parameters
        ----------
        new_img : np.array
            the new image
        
        Raises
        ------
        NotImplementedError
            if not overwritten by subclass
        
        """

        raise NotImplementedError

    @abstractmethod
    def save(self, *args, **kwargs):
        """
        Abstract Function to save image and landmarks
        
        Parameters
        ----------
        *args :
            positional arguments
        **kwargs :
            keyword arguments

        Raises
        ------
        NotImplementedError
            if not overwritten by subclass
        
        """

        raise NotImplementedError

    @abstractmethod
    def save_image(self, *args, **kwargs):
        """
        Abstract Function to save image
        
        Parameters
        ----------
        *args :
            positional arguments
        **kwargs :
            keyword arguments

        Raises
        ------
        NotImplementedError
            if not overwritten by subclass
        
        """
        raise NotImplementedError

    @abstractmethod
    def save_landmarks(self, *args, **kwargs):
        """
        Abstract Function to save landmarks
        
        Parameters
        ----------
        *args :
            positional arguments
        **kwargs :
            keyword arguments

        Raises
        ------
        NotImplementedError
            if not overwritten by subclass
        
        """
        raise NotImplementedError

    @abstractmethod
    def _save_landmarks(self, *args, **kwargs):
        """
        Abstract internal Function to save landmarks
        
        Parameters
        ----------
        *args :
            positional arguments
        **kwargs :
            keyword arguments

        Raises
        ------
        NotImplementedError
            if not overwritten by subclass
        
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def is_gray(self):
        """
        Property returning whether the image is a grayscale image
        
        Raises
        ------
        NotImplementedError
            if not overwritten by subclass
        
        """

        raise NotImplementedError

    @property
    @abstractmethod
    def is_homogeneous(self):
        """
        Property returning whether the landmarks are in homogeneous coordinates
        
        Raises
        ------
        NotImplementedError
            if not overwritten by subclass
        
        """

        raise NotImplementedError

    @is_homogeneous.setter
    @abstractmethod
    def is_homogeneous(self, new_state):
        """
        Setter to update whether the landmarks are in homogeneous coordinates

        Parameters
        ----------
        new_state : bool
            the new value
        
        Raises
        ------
        NotImplementedError
            if not overwritten by subclass
        
        """

        raise NotImplementedError

    @abstractmethod
    def apply_trafo(self, *args, **kwargs):
        """
        Applies a given transformation to image and landmarks

        Parameters
        ----------
        *args :
            positional arguments
        **kwargs :
            keyword arguments
        
        Raises
        ------
        NotImplementedError
            if not overwritten by subclass
        
        """

        raise NotImplementedError

    @abstractmethod
    def _transform_img(self, *args, **kwargs):
        """
        Applies a given transformation to image

        Parameters
        ----------
        *args :
            positional arguments
        **kwargs :
            keyword arguments
        
        Raises
        ------
        NotImplementedError
            if not overwritten by subclass
        
        """
        raise NotImplementedError

    @abstractmethod
    def _transform_lmk(self, *args, **kwargs):
        """
        Applies a given transformation to landmarks

        Parameters
        ----------
        *args :
            positional arguments
        **kwargs :
            keyword arguments
        
        Raises
        ------
        NotImplementedError
            if not overwritten by subclass
        
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_files(cls, *args, **kwargs):
        """
        Creates a class instance from files

        Parameters
        ----------
        *args :
            positional arguments
        **kwargs :
            keyword arguments
        
        Raises
        ------
        NotImplementedError
        
        """

        raise NotImplementedError

    @abstractmethod
    def transform(self, *args, **kwargs):
        """
        Applies a given transformation to image and landmarks

        Parameters
        ----------
        *args :
            positional arguments
        **kwargs :
            keyword arguments
        
        Raises
        ------
        NotImplementedError
            if not overwritten by subclass
        
        """
        raise NotImplementedError

    @abstractmethod
    def cartesian_coordinates(self):
        """
        Transforms the landmarks into cartesian coordinates
        
        Raises
        ------
        NotImplementedError
            if not overwritten by subclass
        
        """
        raise NotImplementedError

    @abstractmethod
    def homogeneous_coordinates(self):
        """
        Transforms the landmarks into homogeneous coordinates
        
        Raises
        ------
        NotImplementedError
            if not overwritten by subclass
        
        """

        raise NotImplementedError

    @abstractmethod
    def transform_about_centre(self, *args, **kwargs):
        """
        Applies a given transformation to image and landmarks at image center
        (internally shifts image and landmarks center to origin, applies 
        transformation and shifts back)

        Parameters
        ----------
        *args :
            positional arguments
        **kwargs :
            keyword arguments
        
        Raises
        ------
        NotImplementedError
            if not overwritten by subclass
        
        """
        raise NotImplementedError

    @abstractmethod
    def resize(self, *args, **kwargs):
        """
        Resizes image and landmarks

        Parameters
        ----------
        *args :
            positional arguments
        **kwargs :
            keyword arguments
        
        Raises
        ------
        NotImplementedError
            if not overwritten by subclass
        
        """
        raise NotImplementedError

    @abstractmethod
    def rescale(self, *args, **kwargs):
        """
        Rescales image and landmarks

        Parameters
        ----------
        *args :
            positional arguments
        **kwargs :
            keyword arguments
        
        Raises
        ------
        NotImplementedError
            if not overwritten by subclass
        
        """
        raise NotImplementedError

    @abstractmethod
    def rotate(self, *args, **kwargs):
        """
        Rotates image and landmarks

        Parameters
        ----------
        *args :
            positional arguments
        **kwargs :
            keyword arguments
        
        Raises
        ------
        NotImplementedError
            if not overwritten by subclass
        
        """
        raise NotImplementedError

    @abstractmethod
    def translate(self, *args, **kwargs):
        """
        Translates image and landmarks

        Parameters
        ----------
        *args :
            positional arguments
        **kwargs :
            keyword arguments
        
        Raises
        ------
        NotImplementedError
            if not overwritten by subclass
        
        """
        raise NotImplementedError

    @abstractmethod
    def view(self, *args, **kwargs):
        """
        Plots image and landmarks

        Parameters
        ----------
        *args :
            positional arguments
        **kwargs :
            keyword arguments
        
        Raises
        ------
        NotImplementedError
            if not overwritten by subclass
        
        """
        raise NotImplementedError

    @abstractmethod
    def normalize_rotation(self, *args, **kwargs):
        """
        Rotates image and landmarks in a way, that the vector between two given 
        points is parallel to horizontal axis

        Parameters
        ----------
        *args :
            positional arguments
        **kwargs :
            keyword arguments
        
        Raises
        ------
        NotImplementedError
            if not overwritten by subclass
        
        """
        raise NotImplementedError

    @abstractmethod
    def _normalize_rotation(self, *args, **kwargs):
        """
        Internal implementation of 
        :meth:`AbstractSingleImage.normalize_rotation`

        Parameters
        ----------
        *args :
            positional arguments
        **kwargs :
            keyword arguments
        
        Raises
        ------
        NotImplementedError
            if not overwritten by subclass
        
        """
        raise NotImplementedError

    @abstractmethod
    def crop(self, *args, **kwargs):
        """
        Crops image and landmarks to given range
        
        Parameters
        ----------
        *args :
            positional arguments
        **kwargs :
            keyword arguments
        
        Raises
        ------
        NotImplementedError
            if not overwritten by subclass
        
        """

        raise NotImplementedError

    @abstractmethod
    def _crop(self, *args, **kwargs):
        """
        Internal implementation of 
        :meth:`AbstractSingleImage.crop`

        Parameters
        ----------
        *args :
            positional arguments
        **kwargs :
            keyword arguments
        
        Raises
        ------
        NotImplementedError
            if not overwritten by subclass
        
        """
        raise NotImplementedError

    @abstractmethod
    def _crop_lmks(self, *args, **kwargs):
        """
        Crops the landmarks

        Parameters
        ----------
        *args :
            positional arguments
        **kwargs :
            keyword arguments
        
        Raises
        ------
        NotImplementedError
            if not overwritten by subclass
        
        """
        raise NotImplementedError

    @abstractmethod
    def crop_to_landmarks(self, *args, **kwargs):
        """
        Crops image and landmarks to bounding box specified by landmarks

        Parameters
        ----------
        *args :
            positional arguments
        **kwargs :
            keyword arguments
        
        Raises
        ------
        NotImplementedError
            if not overwritten by subclass
        
        """
        raise NotImplementedError

    @abstractmethod
    def _crop_to_landmarks(self, *args, **kwargs):
        """
        Internal implementation of 
        :meth:`AbstractSingleImage.crop_to_landmarks`

        Parameters
        ----------
        *args :
            positional arguments
        **kwargs :
            keyword arguments
        
        Raises
        ------
        NotImplementedError
            if not overwritten by subclass
        
        """
        raise NotImplementedError

    @abstractmethod
    def get_landmark_bounds(self, *args, **kwargs):
        """
        Calculates bounds of landmarks

        Parameters
        ----------
        *args :
            positional arguments
        **kwargs :
            keyword arguments
        
        Raises
        ------
        NotImplementedError
            if not overwritten by subclass
        
        """
        raise NotImplementedError

    @abstractmethod
    def to_grayscale(self):
        """
        Converts image to grayscale
        
        Raises
        ------
        NotImplementedError
            if not overwritten by subclass
        
        """

        raise NotImplementedError


class BaseSingleImage(AbstractSingleImage):
    """
    Holds Single Image

    """

    def __init__(self, img: np.ndarray, *args, **kwargs):
        super().__init__()

        self.img = img


    @property
    def img(self):
        return self._img

    @img.setter
    def img(self, image):

        # create image channel if necessary
        if len(image.shape) < 3:
            image = image.reshape(*image.shape, 1)

        # ensure channels at back
        if image.shape[0] == 1 or image.shape[0] == 3:
            image = image.transpose((*range(1, len(image.shape)), 0))

        self._img = image

    def save(self, directory, filename, lmk_type="LJSON", **kwargs):
        """
        Saves Image and optionally landmarks to files

        Parameters
        ----------
        directory : str
            string containing the directory to save
        filename : str
            string containing the filename (without the extension)
        lmk_type : str or None
            if None: no landmarks will be saved
            if str: specifies type of landmark file
        **kwargs :
            additional keyword arguments passed to save function for landmarks

        """
        self.save_image(os.path.join(directory, filename + ".png"))

        if lmk_type is not None:
            self.save_landmarks(os.path.join(directory, filename), lmk_type,
                                **kwargs)

    def save_image(self, filepath):
        """
        Saves Image to file

        Parameters
        ----------
        filepath : str
            file to save the image to

        """
        imsave(filepath, self.img.squeeze())

    def save_landmarks(self, filepath, lmk_type="LJSON", **kwargs):
        """
        Saves landmarks to file

        Parameters
        ----------
        filepath : str
            path to file the landmarks should be saved to
        lmk_type : str
            specifies the type of landmark file
        **kwargs :
            additional keyword arguments passed to save function

        """
        self._save_landmarks(filepath, lmk_type, **kwargs)

    @abstractmethod
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
        NotImplementedError
            If not overwritten by subclass

        """
        raise NotImplementedError

    @property
    def is_gray(self):
        return self.img.shape[-1] == 1

    @property
    def is_homogeneous(self):
        return not self.is_cartesian

    @is_homogeneous.setter
    def is_homogeneous(self, homogeneous: bool):
        self.is_cartesian = not homogeneous

    def apply_trafo(self, transformation: AffineTransform, **kwargs):
        """
        Apply transformation inplace to image and landmarks

        Parameters
        ----------
        transformation : :class:`skimage.transform.AffineTransform`
            transformation to apply
        **kwargs :
            additional keyword arguments

        Returns
        -------
        :class:`BaseSingleImage`
            Transformed Image and Landmarks

        """

        # ensure transformation to be affine
        transformation = AffineTransform(transformation.params)

        self._transformation_history.append(transformation)
        self._transform_img(transformation, **kwargs)

        self._transform_lmk(transformation)

        return self

    def _transform_img(self, transformation: AffineTransform, **kwargs):
        """
        Apply transformation inplace to image

        Parameters
        ----------
        transformation : :class:`skimage.transform.AffineTransform`
            transformation to apply
        **kwargs :
            additional keyword arguments

        Returns
        -------
        :class:`BaseSingleImage`
            Transformed Image with original Landmarks

        """

        self.img = warp(np.ascontiguousarray(self.img), transformation.inverse,
                        **kwargs)

        return self

    @abstractmethod
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
        raise NotImplementedError

    @classmethod
    def from_files(cls, file, extension=None, **kwargs):

        file = os.path.abspath(file)

        if not extension:
            # potential file are all files in same directory whose name starts
            # with the files name without the extension
            potential_files = [os.path.join(os.path.split(file)[0], x)
                               for x in os.listdir(os.path.split(file)[0])
                               if x.startswith(os.path.split(file)[1].rsplit(
                    ".", 1)[0])]

            if any([_file.endswith(".ljson") for _file in
                    potential_files]):
                extension = ".ljson"
            elif any([_file.endswith(".pts") for _file in
                      potential_files]):
                extension = ".pts"

            else:
                extension = ".txt"

        if extension == ".ljson":
            return cls.from_ljson_files(file, **kwargs)
        elif extension == ".pts":
            return cls.from_pts_files(file, **kwargs)
        else:
            return cls.from_npy_files(file, **kwargs)

    @classmethod
    @abstractmethod
    def from_npy_files(cls, file, **kwargs):
        """
        Create class from image or landmark file

        Parameters
        ----------
        file : str
            path to image or landmarkfile

        Returns
        -------
        :class:`BaseSingleImage`

        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_pts_files(cls, file, **kwargs):
        """
        Create class from image or landmark file
        Parameters
        ----------
        file: string
            path to image or landmarkfile

        Returns
        -------
        :class:`BaseSingleImage`

        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_ljson_files(cls, img_file, **kwargs):
        """
        Create class from image or landmark file

        Parameters
        ----------
        file: str
            path to image or landmarkfile

        Returns
        -------
        :class:`BaseSingleImage`

        """
        raise NotImplementedError

    def transform(self, transform=None, rotation=None, scale=None, translation=None,
                  shear=None, trafo_matrix=None,
                  return_matrix=False, **kwargs):
        """
        transform image and landmarks by parameters or transformation matrix
        See :class:`skimage.transform.AffineTransform` for a detailed parameter
        explanation

        Parameters
        ----------
        transform : :class:`skimage.transform.AffineTransform`
            if transform is specified it overwrites all other arguments
        rotation : float or None
            rotation angle in radiant
        scale : float or None
            scale value
        translation : 
            translation params
        shear :
            shear params
        trafo_matrix :
            transformation matrix
        return_matrix : bool
            whether to return the transformation matrix along the transformed
            object
        **kwargs :
            additional keyword arguments

        Returns
        -------
        :class:`BaseSingleImage`
            transformed Image
        [optional] np.ndarray
            transformation matrix

        """
        new_instance = deepcopy(self)

        if transform is None:
            trafo = AffineTransform(rotation=rotation, scale=scale,
                                    translation=translation, shear=shear,
                                    matrix=trafo_matrix)
        else:
            trafo = transform

        new_instance.apply_trafo(trafo, **kwargs)

        if return_matrix:
            return new_instance, trafo.params
        else:
            return new_instance

    @abstractmethod
    def cartesian_coordinates(self):
        """
        Transforms landmark coordinates inplace to cartesian coordinates

        Returns
        -------
        :class:`BaseSingleImage`
            Image with Landmarks in cartesian Coordinates

        """
        raise NotImplementedError

    @abstractmethod
    def homogeneous_coordinates(self):
        """
        Transforms landmark coordinates inplace to homogeneous coordinates

        Returns
        -------
        :class:`BaseSingleImage`
            Image with Landmarks in Homogeneous Coordinates

        """
        raise NotImplementedError

    def transform_about_centre(self, transform=None, rotation=None, scale=None,
                               translation=None, shear=None, trafo_matrix=None,
                               return_matrix=False, **kwargs):
        """
        Perform transformations about the image center.
        (internally shifting image to origin, perform transformation and
        shift it back)

        Parameters
        ----------
        transform : :class:`skimage.transform.AffineTransform`
            if transform is specified it overwrites all other arguments
        rotation : float
            rotation angle in radiant
        scale : float
            scale value
        translation :
            translation params
        shear :
            shear params
        trafo_matrix :
            transformation matrix
        return_matrix : bool
            whether to return the transformation matrix along the transformed
            object
        **kwargs :
            additional keyword arguments
        
        Returns
        -------
        :class:`BaseSingleImage`
            transformed Image
        [optional] np.ndarray
            transformation matrix

        """
        if transform is None:

            affine_trafo = AffineTransform(rotation=rotation, scale=scale,
                                           translation=translation,
                                           shear=shear, matrix=trafo_matrix)
        else:
            affine_trafo = transform

        shift_y, shift_x = np.array(self.img.shape[:2]) / 2.

        # transform to shift image to origin
        tf_shift = AffineTransform(translation=[-shift_x, -shift_y])

        # transform to shift image back to original position
        tf_shift_inv = AffineTransform(translation=[shift_x, shift_y])

        complete_trafo = (tf_shift + (affine_trafo + tf_shift_inv))

        return self.transform(transform=complete_trafo,
                              return_matrix=return_matrix, **kwargs)

    def resize(self, target_shape, **kwargs):
        """
        resize image and scale landmarks
        Parameters
        ----------
        target_shape : tuple or list
            target shape for resizing
        **kwargs :
            additional keyword arguments (passed to 
            :meth:`skimage.transform.warp`)

        Returns
        -------
        :class:`BaseSingleImage`
            transformed Image

        """
        scale = np.asarray(target_shape) / np.asarray(self.img.shape[:-1])
        scale = np.array([scale[1], scale[0]])

        return self.transform(scale=scale, output_shape=target_shape, **kwargs)

    def rescale(self, scale, **kwargs):
        """
        Scale Image and landmarks

        Parameters
        ----------
        scale :
            scale parameter
        **kwargs :
            additional keyword arguments (passed to 
            :meth:`skimage.transform.warp`)

        Returns
        -------
        :class:`BaseSingleImage`
            transformed Image

        """
        target_shape = np.asarray(self.img.shape[:-1]) * np.asarray(scale)

        return self.transform(scale=scale, output_shape=target_shape, **kwargs)

    def rotate(self, angle, degree=True, **kwargs):
        """
        Rotates the image and landmarks by given angle

        Parameters
        ----------
        angle : float or int
            rotation angle
        degree : bool
            whether the angle is given in degree or radiant
        **kwargs :
            additional keyword arguments (passed to 
            :meth:`skimage.transform.warp`)

        Returns
        -------
        :class:`BaseSingleImage`
            transformed Image

        """

        if degree:
            angle = np.deg2rad(angle)

        return self.transform_about_centre(rotation=angle, **kwargs)

    def translate(self, translation, relative=False, **kwargs):
        """
        translates image and landmarks

        Parameters
        ----------
        translation :
            translation parameters
        relative : bool
            whether translation parameters are relative to image size
       **kwargs :
            additional keyword arguments (passed to 
            :meth:`skimage.transform.warp`)

        Returns
        -------
        :class:`BaseSingleImage`
            transformed Image

        """
        if relative:
            translation = translation * self.img.shape[:-1]

        return self.transform(translation=translation, **kwargs)

    @abstractmethod
    def view(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def normalize_rotation(self, *args, **kwargs):
        raise NotImplementedError

    def _normalize_rotation(self, lmks, index_left, index_right, **kwargs):
        """
        normalizes rotation based on two keypoints

        Parameters
        ----------
        lmks : np.ndarray
            landmarks for rotation normalization
        index_left : int
            index for left point
        index_right : int
            index for right point
        **kwargs :
            additional keyword arguments (passed to 
            :meth:`skimage.transform.warp`)

        Returns
        -------
        :class:`BaseSingleImage`
            transformed Image

        """
        left = lmks[index_left]
        right = lmks[index_right]

        def get_angle(v0, v1, v2, degree=False):
            """
            Calculate the angle between v1 and v2 with v0 as anchor point

            Parameters
            ----------
            v0 : np.array
                Anchor point
            v1 : np.array
                First vector
            v2 : np.array
                Second Vector
            degree : bool
                if True: returns angle in degree, else in rad

            Returns
            -------
            angle

            """

            a1 = v0 - v1
            a2 = v0 - v2
            cosine_angle = np.dot(a1, a2) / (np.linalg.norm(a1) *
                                             np.linalg.norm(a2))

            angle = np.arccos(cosine_angle)

            if degree:
                angle = np.rad2deg(angle)

            return angle

        diff = left - right

        middle = right + diff / 2

        length_middle_left = np.sqrt(((left - middle) ** 2).sum())

        left_optim = deepcopy(middle)
        left_optim[-1] += length_middle_left

        rot_angle = get_angle(middle, left_optim, left, degree=False)
        return self.transform_about_centre(rotation=rot_angle, **kwargs)

    def crop(self, min_y, min_x, max_y, max_x):
        """
        Crops Image by specified values

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
        :class:`BaseSingleImage`
            cropped image

        """

        # ensure cropping values are withing the image bounds
        # else set cropping val to image bound
        min_y, min_x = max(0, min_y), max(0, min_x)
        max_y, max_x = min(self.img.shape[0], max_y), \
                       min(self.img.shape[1], max_x)

        return deepcopy(self)._crop(int(np.floor(min_y)), int(np.floor(min_x)),
                                    int(np.ceil(max_y)),
                                    int(np.ceil(max_x)))

    @abstractmethod
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

        Raises
        -------
        NotImplementedError
            if not overwritten by subclass

        """
        raise NotImplementedError

    @staticmethod
    def _crop_lmks(lmks, min_y, min_x, max_y, max_x):
        """
        Crops landmarks to given values

        Parameters
        ----------
        lmks : np.ndarray
            landmarks to crop
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
        np.ndarray
            cropped landmarks
        
        """
        # lmk_mask_y = (lmks[:, 0] >= min_y) & (lmks[:, 0] <= max_y)
        # lmk_mask_x = (lmks[:, 1] >= min_x) & (lmks[:, 1] <= max_x)
        #
        # lmk_mask = lmk_mask_x & lmk_mask_y
        #
        # return lmks[lmk_mask] - np.array((min_y, min_x))
        return lmks - np.array((min_y, min_x))

    def crop_to_landmarks(self, proportion=0., **kwargs):
        """
        Crop image to landmarks

        Parameters
        ----------
        proportion : float
            image proportion to add to size of bounding box
        **kwargs :
            additional keyword arguments

        Returns
        -------
        :class:`BaseSingleImage`
            cropped image

        """
        return self._crop_to_landmarks(proportion, **kwargs)

    @abstractmethod
    def _crop_to_landmarks(self, proportion=0., **kwargs):
        """
        Crop to landmarks inplace

        Parameters
        ----------
        proportion : float
            boundary proportion of cropping
        **kwargs :
            additional keyword arguments

        Raises
        ------
        NotImplementedError
            if not overwritten by subclass

        """
        raise NotImplementedError

    @staticmethod
    def get_landmark_bounds(lmks):
        """
        Function to calculate the landmark bounds

        Parameters
        ----------
        lmks : np.ndarray
            landmarks

        Returns
        -------
        int: min_y
        int: min_x
        int: max_y
        int: max_x
        """

        min_y = lmks[:, 0].min()
        max_y = lmks[:, 0].max()
        min_x = lmks[:, 1].min()
        max_x = lmks[:, 1].max()

        return min_y, min_x, max_y, max_x

    def to_grayscale(self):
        """

        Convert Image to grayscale

        Returns
        -------
        :class:`BaseSingleImage`
            Grayscale Image

        """

        new_instance = deepcopy(self)

        if not new_instance.is_gray:
            new_instance.img = rgb2gray(self.img).reshape(
                *self.img.shape[:-1], 1)

        return new_instance
