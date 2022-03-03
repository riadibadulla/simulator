import abc
import copy
import pickle

import astropy.units as u
import numpy as np

from . import logging
from . import plotting
from . import utils


class Distribution(logging.LoggerMixin, plotting.Plottable):
    """Abstract class representing a 2D distribution. """

    @u.quantity_input(wavelength=u.m, pixel_scale=u.m)
    def __init__(self, wavelength, pixel_scale, npix):
        self.wavelength = wavelength
        self.pixel_scale = pixel_scale
        self.npix = npix
        self.size = npix * pixel_scale
        self.k = np.pi * 2.0 / self.wavelength
        self.x, self.y = utils.mesh_grid(npix, pixel_scale)

    def __repr__(self):
        return f"{self.__class__.__name__} " \
            f"of wavelength = {self.wavelength:.2f}, " \
            f"pixel scale = {self.pixel_scale:.2f}, " \
            f"size {self.npix} x {self.npix}."

    def copy(self):
        """A method to *deep* copy the distribution.

        :return: a copy of the wavefront
        :rtype: pyoptica.Wavefront

        **Example**

        >>> import astropy.units as u
        >>> import pyoptica as po
        >>>
        >>> wavelength = 500 * u.nm
        >>> pixel_scale = 22 * u.um
        >>> npix = 1024
        # As an example we use a Wavefront. Distribution is an abstractclass!
        >>> wf = po.Wavefront(wavelength, pixel_scale, npix)
        >>> new_wf = wf.copy()

        """
        return copy.deepcopy(self)

    def to_pickle(self, path):
        """A method to serialize distribution to pickle.

        :param path: path to the file to which distribution is dumped
        :type path: str

        **Example**

        >>> import astropy.units as u
        >>> import pyoptica as po
        >>>
        >>> wavelength = 500 * u.nm
        >>> pixel_scale = 22 * u.um
        >>> npix = 1024
        >>> # We are going to work on `Image` as `Distribution` is abstract
        >>> image = Image(wavelength, pixel_scale, npix)
        >>> file_path = 'path/to/your/file.pickle'
        >>> image.to_pickle(file_path)

        """
        with open(path, 'wb') as file:
            pickle.dump(self, file, pickle.HIGHEST_PROTOCOL)
        self.logger.debug(f"Saved {self.__class__.__name__} to {path}")

    @classmethod
    def from_pickle(cls, path):
        """A method to load distribution serialized to pickle.

        :param path: path to the file to be loaded.
        :type path: str
        :returns: loaded distribution
        :rtype: pyoptica.Distribution (or any of the subclasses)

        **Example**

        >>> import pyoptica as po
        >>>
        >>> file_path = 'path/to/your/file.pickle'
        >>> wf = po.Wavefront.from_pickle(file_path)

        """
        with open(path, 'rb') as file:
            wf = pickle.load(file)
        cls.logger.debug(f"Loaded {cls.__class__.__name__} from {path}")
        return wf

    @abc.abstractmethod
    def to_fits(self, path):
        pass

    @classmethod
    @abc.abstractmethod
    def from_fits(cls, path):
        pass
