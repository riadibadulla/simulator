import logging
import warnings

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pkg_resources
import torch


def fft(array):
    """Performs 2D FFT for array with given method

    :param array: array to be Fourier transformed
    :type array: np.array
    :param method: description of algorithm (package)
    :type method: str
    :return: Fourier transformed array
    :rtype: np.array

    **Example**

    >>> import pyoptica as po
    >>>
    >>> arr = np.ones((1024, 1024))
    >>> po.utils.fft(arr)

    """
    return torch.fft.fft2(array)


def ifft(array: np.array) -> np.array:
    """Performs 2D IFFT for array with given method

    :param array: array to be inversely Fourier transformed
    :type array: np.array
    :param method: description of algorithm (package)
    :type method: str
    :return: Fourier transformed array
    :rtype: np.array

    **Example**

    >>> import pyoptica as po
    >>>
    >>> arr = np.ones((1024, 1024))
    >>> po.utils.ifft(arr)

    """
    return torch.fft.ifft2(array)


def mesh_grid(npix, pixel_scale):
    """Create a grid of size `npix` with sampling defined by `pixel_scale`
    centered at 0,0.

    :param npix: Size of the grid (in each direction).
    :type npix: int
    :param pixel_scale: Sampling of the grid
    :type pixel_scale: astropy.Quantity (type is not forced!)
    :return: two dimensional mesh grid (x, y)
    :rtype: (np.array, np.array)

    **Example**

    >>> import astropy.units as u
    >>> import pyoptica as po
    >>>
    >>> pixel_scale = 1 * u.um
    >>> npix = 512
    >>> x, y = po.utils.mesh_grid(npix, pixel_scale)

    """
    y, x = (np.indices((npix, npix)) - npix / 2) * pixel_scale
    # return torch.tensor(x), torch.tensor(y)
    return x,y


def rgb2gray(rgb):
    """Converts rgb(a) image to grayscale. gray = 0.299r + 0.587g + 0.114b
    Based on: https://en.wikipedia.org/wiki/Grayscale

    :param rgb: array representing an image rgb(a) of size (n, m, 3/(4))
    :type: np.array
    :return: grayscale image of size (n, m)
    :rtype: np.array

    **Example**

    >>> import pyoptica as po
    >>> rgb = np.ones((16, 16, 3))
    >>> po.utils.rgb2gray(rgb)
    """
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

