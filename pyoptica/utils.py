import logging
import warnings

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pkg_resources
import torch

_log = logging.getLogger(__name__)
_log.setLevel(logging.DEBUG)
_ch = logging.StreamHandler()
_ch.setLevel(logging.DEBUG)
_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
_ch.setFormatter(_formatter)
_log.addHandler(_ch)

LOGOS = dict(
    round=pkg_resources.resource_filename(
        __name__, 'data/logos/pyoptica_round.png'),
    rect=pkg_resources.resource_filename(
        __name__, 'data/logos/pyoptica_rect.png'),
    text=pkg_resources.resource_filename(
        __name__, 'data/logos/pyoptica_text_only.png'),
)


def cart2pol(x, y):
    """Calculates polar coordinates from given cartesian coordinates.

    :param x: X coordinate
    :type x: float
    :param y: Y coordinate
    :type y: float
    :return: Polar coordinates of given point
    :rtype: tuple[float, float]

    **Example**

    >>> import pyoptica as po
    >>>
    >>> x, y = 1 * u.m, 1 * u.m
    >>> po.utils.cart2pol(x, y)
    (<Quantity 1.41421356 m>, <Quantity 0.78539816 rad>)

    """

    rho = np.hypot(x, y)
    phi = np.arctan2(y, x)
    return rho, phi


def pol2cart(rho, phi):
    """Calculates cartesian coordinates from given polar coordinates.

    :param rho: Radial coordinate
    :type rho: float
    :param phi: Angular coordinate
    :type phi: float
    :return: Cartesian coordinates of given point
    :rtype: tuple[float, float]

    **Example**

    >>> import pyoptica as po
    >>>
    >>> r, p = 1 * u.m, 1 * u.rad
    >>> po.utils.pol2cart(r, p)
    (<Quantity 0.54030231 m>, <Quantity 0.84147098 m>)
    """

    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y


def fft(array, method='numpy'):
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
    if method == 'numpy':
        ishifted_array = torch.fft.ifftshift(array)
        fft_array = torch.fft.fft2(ishifted_array)
        shifted_fft_array = torch.fft.fftshift(fft_array)
        return shifted_fft_array
    else:
        raise ValueError(f"")


def ifft(array: np.array, method='numpy') -> np.array:
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
    if method == 'numpy':
        ishifted_array = torch.fft.ifftshift(array)
        ifft_array = torch.fft.ifft2(ishifted_array)
        shifted_ifft_array = torch.fft.ifftshift(ifft_array)
        return shifted_ifft_array
    else:
        raise ValueError(f"")


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


def space2freq(x):
    """Converts space coordinates into frequency, x->f, y->g
    :param x: X grid
    :type x: astropy.Quantity array of type length
    :param y: Y grid
    :type y: astropy.Quantity array of type length
    :return: (array, array)

    """
    pixel_scale = x[0, 0] - x[1, 1]
    npix = x.shape[0]
    denominator = pixel_scale ** 2 * npix
    f = x / denominator
    return f


def range_validator(value, range):
    """Checks if value is within specified range and clamps it according to
    given boundaries

    :param value: Numeric value to check if it's range is valid.
    :type value: float
    :param range: Tuple with lower and upper limit for checked value.
    :type range: (float, float)
    :return: Given value or clamped to the given range.
    :rtype: float

    """
    min_val, max_val = range
    if value < min_val:
        warning = f"Value out of valid range: {range}, return value set to: {min_val}"
        warnings.warn(warning)
        _log.warning(warning)
        return min_val
    elif value > max_val:
        warning = f"Value out of valid range: {range}, return value set to: {max_val}"
        warnings.warn(warning)
        _log.warning(warning)
        return max_val
    else:
        return value


def param_validator(param, params):
    """Checks if parameter is listed in accepted parameters list, if not returns
    first listed valid parameter.

    :param param: Parameter to be check if is listed in valid parameters list.
    :type param: object
    :param params: List of valid parameters
    :type params: list of objects
    :return: Given parameter or first listed as valid.
    :rtype: float

    """
    if param not in params:
        warning = f"Parameter: {param} is not listed as valid: {params}, param set to default: {params[0]}"
        warnings.warn(warning)
        _log.warning(warning)
        return params[0]
    else:
        return param


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


def get_logo(kind='round'):
    """Loads PyOptica logo as a numpy array. `rect`, `round`, and `text` kinds
    are available.

    :param kind: logo kind (`rect`, `round`, `text`)
    :type kind: str
    :return: loaded logo
    :rtype: np.array
    :raises: KeyError

    **Example**
    >>> import pyoptica as po
    >>> po.utils.get_logo()
    """
    try:
        path = LOGOS[kind]
    except KeyError as e:
        raise KeyError(f'{kind} is not supported.'
                       f' Only {list(LOGOS.keys())} are supported') from e
    img = plt.imread(path)
    img_gray = rgb2gray(img)
    return img_gray
