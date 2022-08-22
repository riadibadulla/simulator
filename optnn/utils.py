import numpy as np
import torch


def fft(array):
    """Performs 2D FFT for array with given method

    :param array: array to be Fourier transformed
    :type array: np.array
    :param method: description of algorithm (package)
    :type method: str
    :return: Fourier transformed array
    :rtype: np.array
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
    """
    y, x = (np.indices((npix, npix)) - npix / 2) * pixel_scale
    return x,y



