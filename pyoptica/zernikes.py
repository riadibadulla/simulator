from math import ceil, factorial, floor, sqrt

import astropy.units as u
import numpy as np
from scipy.optimize import minimize

from . import logging

__all__ = [
    'zernike', 'fit_zernikes', 'fit_zernikes_lstsq', 'construct_wavefront',
    'get_zernike_name', 'mn_to_noll', 'mn_to_fringe', 'mn_to_osa',
    'noll_to_mn', 'osa_to_mn', 'fringe_to_mn'
]

logger = logging.get_standard_logger(__name__)

ZERNIKE_NAMES = {
    # Based on *Wavefront Optics for Vision Correction* by Guang-ming Dai
    (0, 0): 'Piston', (-1, 1): 'y-Tilt', (1, 1): 'x-Tilt',
    (-2, 2): 'y-Astigmatism', (0, 2): 'Defocus', (2, 2): 'x-Astigmatism',
    (-3, 3): 'y-Trefoil', (-1, 3): 'y-Coma', (1, 3): 'x-Coma',
    (3, 3): 'x-Trefoil', (-4, 4): 'y-Quadrafoil',
    (-2, 4): 'y-Secondary Astigmatism', (0, 4): 'Spherical Aberration',
    (2, 4): 'x-Secondary Astigmatism', (4, 4): 'x-Quadrafoil'
}  # I use mn to avoid any confusion about indexing!


def zernike(index, convention, rho, theta, normalize=True, fill_value=0):
    """Calculates zernike polynomial for given index (or indices) for
    convention (`mn`, `noll`, `osa`). Implementation based on [1]. If
     normalize is `True` output is normalized following Noll's convention [2].

    :param index: index (or indices for mn) of the polynomial
    :type index: int or tuple of ints
    :param convention: used indexing convention
    :type convention: str
    :param rho: radial coordinates of the plane,
    :type rho: numpy.array
    :param theta: angle coordinate of the plane
    :type theta: numpy.array
    :param normalize: Should output be normalized following Noll's convention?
    :type normalize: bool
    :param fill_value: what value should be put for r > 1
    :type fill_value: numeric
    :return: resulting _zernike_mn distribution
    :rtype: numpy.array


    **Example**

    >>> import astropy.units as u
    >>> import pyoptica as po
    >>>
    >>> npix = 101
    >>> pixel_scale = 0.5 * u.m
    >>> x, y = po.utils.mesh_grid(npix, pixel_scale)
    >>> r_max = .5 * npix * pixel_scale
    >>> r, theta = po.utils.cart2pol(x, y)
    >>> r = r / r_max
    >>> j = 8
    >>> z_8 = po.zernikes.zernike(j, 'noll', r, theta)

    In case we want to try different convention:

    >>> z_33 = po.zernikes.zernike((3, 3), 'mn', r, theta)

    **References**

    [1] Vasudevan Lakshminarayanan and Andre Fleck (2011) -
    *Zernike polynomials: a guide*, Journal of Modern Optics
    
    [2] Robert J. Noll (1976) -
    "*Zernike polynomials and atmospheric turbulence*," J. Opt. Soc. Am. 66, 207-211

    """
    m, n = _to_mn(index, convention)
    return _zernike_mn(m, n, rho, theta, normalize, fill_value)


def _zernike_mn(m, n, rho, theta, normalize=True, fill_value=0):
    """Calculates zernike polynomial for given indices m, n for rho and theta.
    Implementation based on [1]. If normalize is `True` output is normalized
    following Noll's convention [2].

    :param m: zernike m-degree, angular frequency
    :type m: int
    :param n: zernike n-degree, radial order
    :type n: int
    :param rho: radial coordinates of the plane,
    :type rho: numpy.array
    :param theta: angle coordinate of the plane
    :type theta: numpy.array
    :param normalize: Should output be normalized following Noll's convention?
    :type normalize: bool
    :param fill_value: what value should be put for r > 1
    :type fill_value: numeric
    :return: resulting _zernike_mn distribution
    :rtype: numpy.array

    **References**

    [1] Vasudevan Lakshminarayanan and Andre Fleck (2011) -
    *Zernike polynomials: a guide*, Journal of Modern Optics

    [2] Robert J. Noll (1976) -
    "*Zernike polynomials and atmospheric turbulence*," J. Opt. Soc. Am. 66, 207-211

    """
    if n < 0:
        raise ValueError(f"Radial order n = {n} < 0! Must be n > 0.")
    if m > n:
        raise ValueError(f"Radial order n > m angular frequency: {n} > {m}.")
    if (n - m) % 2 != 0:
        raise ValueError(f"Radial order n - m angular frequency is not even: "
                         f"{n} - {m} = {n - m}")
    if isinstance(rho, u.Quantity):
        rho = rho.value
    if isinstance(theta, u.Quantity):
        theta = theta.value
    flat_disk = np.ones_like(rho)
    flat_disk[rho > 1] = fill_value
    radial_function = R(m, n, rho) * flat_disk
    if m >= 0:
        output = radial_function * np.cos(m * theta)
    else:
        output = radial_function * np.sin(-m * theta)

    if normalize:
        output *= norm_coefficient(m, n)

    return output


def R(m, n, rho):
    r"""The radial function described by:

    :math: `R_{n}^{m}(r)=\sum_{l=0}^{(n-m) / 2} \frac{(-1)^{l}(n-l) !}{l !\left[\frac{1}{2}(n+m)-l\right] !\left[\frac{1}{2}(n-m)-l\right] !} r^{n-2 l}`


    The definition was taken from [1], however, good description of zernikes
    can be found in almost all textbooks.

    :param m: m-degree, angular frequency
    :type m: int
    :param n: n-degree, radial order
    :type n: int
    :param rho: r coordinates of the plane,
    :type rho: numpy.array
    :return: calculated radial function
    :rtype: numpy.array

    **Example**

    >>> import astropy.units as u
    >>> import pyoptica as po
    >>>
    >>> npix = 101
    >>> pixel_scale = 0.5 * u.m
    >>> x, y = po.utils.mesh_grid(npix, pixel_scale)
    >>> r_max = .5 * npix * pixel_scale
    >>> r, _ = po.utils.cart2pol(x, y)
    >>> r = r / r_max
    >>> m, n =  1, 3
    >>> radial_poly = po.zernikes.R(m, n, r)

    **References**

    [1] Vasudevan Lakshminarayanan and Andre Fleck (2011) -
    *Zernike polynomials: a guide*, Journal of Modern Optics

    """

    m = abs(m)  # For Zm and Z-m, Rm is the same
    range_stop = (n - m) // 2 + 1
    output = np.zeros_like(rho)
    for l in range(range_stop):
        numerator = (-1.) ** l * factorial(n - l)
        denominator = factorial(l)
        denominator *= factorial((n + m) / 2. - l)
        denominator *= factorial((n - m) / 2. - l)
        output += numerator / denominator * rho ** (n - 2. * l)
    return output


def norm_coefficient(m, n):
    """Calculate normalizaiton coeficient of each zernikes; Rn+-m(1) = 1 for
    all n, m. THhe folowing euqtion is used (eq. 4 in [1]):

    :math:`N_{n}^{m}=\left(\frac{2(n+1)}{1+\delta_{m 0}}\right)^{1 / 2}`

    :param m: m-degree, angular frequency
    :type m: int
    :param n: n-degree, radial order
    :type n: int
    :return: calculate normalization coefficient
    :rtype: float

    **Example**

    >>> import pyoptica as po
    >>>
    >>> m, n = 1, 3
    >>> po.zernikes.norm_coefficient(m, n)
    2.8284271247461903

    **References**

    [1] Vasudevan Lakshminarayanan and Andre Fleck (2011) -
    *Zernike polynomials: a guide*, Journal of Modern Optics

    """
    numerator = 2 * (n + 1)
    denominator = 2 if m == 0 else 1
    norm = (numerator / denominator) ** 0.5
    return norm


def mn_to_osa(m, n):
    """Conversion of m, n indices to OSA j index [1]

    :param m: m-degree, angular frequency
    :type m: int
    :param n: n-degree, radial order
    :type n: int
    :return: j-index
    :rtype: int

    **Example**

    >>> import pyoptica as po
    >>>
    >>> m, n = 1, 3
    >>> po.zernikes.mn_to_osa(m, n)
    8

    **References**

    [1] Vasudevan Lakshminarayanan and Andre Fleck (2011) -
    *Zernike polynomials: a guide*, Journal of Modern Optics

    """
    j = (n * (n + 2) + m) // 2
    logger.debug(f"Converted (m={m}, n={n}) -> j={j}.")
    return j


def osa_to_mn(j):
    """Conversion of OSA j index to m, n [1]

    :param j: j index (OSA convention)
    :type j: int
    :return: m, n - angular frequency and radial order
    :rtype: Tuple(int, int)

    **Example**

    >>> import pyoptica as po
    >>>
    >>> j = 8
    >>> po.zernikes.osa_to_mn(j)
    (1, 3)

    **References**

    [1] Vasudevan Lakshminarayanan and Andre Fleck (2011) -
    *Zernike polynomials: a guide*, Journal of Modern Optics

    """
    n = ceil((-3 + sqrt((9 + 8 * j))) / 2)
    m = 2 * j - n * (n + 2)
    logger.debug(f"Converted j={j} -> (m={m}, n={n}).")
    return m, n


def mn_to_noll(m, n):
    """Conversion of m, n indices to Noll j index [1].

    :param m: m-degree, angular frequency
    :type m: int
    :param n: n-degree, radial order
    :type n: int
    :return: j-index
    :rtype: int

    **Example**

    >>> import pyoptica as po
    >>>
    >>> m, n = 1, 3
    >>> po.zernikes.mn_to_noll(m, n)
    8

    **References**

    [1] https://en.wikipedia.org/wiki/Zernike_polynomials

    """
    first = (n * (n + 1)) // 2
    second = abs(m)
    third = 0
    # I wanted to simplify the logic but it seems to work like a colander
    if m > 0 and n % 4 in [0, 1]:
        third = 0
    elif m < 0 and n % 4 in [2, 3]:
        third = 0
    elif m >= 0 and n % 4 in [2, 3]:
        third = 1
    elif m <= 0 and n % 4 in [0, 1]:
        third = 1
    j = first + second + third
    logger.debug(f"Converted (m={m}, n={n}) -> j={j}.")
    return j


def noll_to_mn(j):
    """Conversion of Noll j index to m, n [1]

    :param j: j index (Noll convention)
    :type j: int
    :return: m, n - angular frequency and radial order
    :rtype: Tuple(int, int)

    **Example**

    >>> import pyoptica as po
    >>>
    >>> j = 8
    >>> po.zernikes.noll_to_mn(j)
    (1, 3)

    **References**

    [1] Alfred K. K. Wong (2005) -
    *Optical Imaging in Projection Microlithography*, SPIE Press

    """
    n = ceil((-3 + sqrt((1 + 8 * j))) / 2)
    m = n - 2 * floor(((n + 2) * (n + 1) - 2 * j) / 4)
    if j & 1:  # is odd?
        m *= -1
    logger.debug(f"Converted j={j} -> (m={m}, n={n}).")
    return m, n


def fringe_to_mn(j):
    """Conversion of Fringe j index to m, n

    :param j: j index (Fringe convention)
    :type j: int
    :return: m, n - angular frequency and radial order
    :rtype: Tuple(int, int)

    **Example**

    >>> import pyoptica as po
    >>>
    >>> j = 8
    >>> po.zernikes.fringe_to_mn(j)
    (-1, 3)
    """
    # How did I get here? I couldn't get the equation anywhere on the Internet.
    # Let me explain you my train of thoughts. If you look closely at this:
    # https://www.spiedigitallibrary.org/ContentImages/Journals/JMMMGF/8/3/031404/WebImages/031404_1_2.jpg
    # You can clearly see that finding m (based on the row) is not a challenge
    # (just make sure that m is of correct sign). Once you know m just use
    # the equation from Wikipedia: all you have to do is solve a quadratic
    # equation given that n>=0.
    row = ceil(sqrt(j))
    m = (((row ** 2 - j) + 1) // 2)
    if (row ** 2 - j) % 2 != 0:
        m *= -1
    if m >= 0:
        n = int(2 * sqrt(2 * m + j) - m - 2)
    else:
        n = int(m - 2 + 2 * sqrt(j - 2 * m - 1))
    return m, n


def mn_to_fringe(m, n):
    """Conversion of Fringe j index to m, n [1]

    :param m: m-degree, angular frequency
    :type m: int
    :param n: n-degree, radial order
    :type n: int
    :return: j-index
    :rtype: int

    **Example**

    >>> import pyoptica as po
    >>>
    >>> m, n = -1, 3
    >>> po.zernikes.mn_to_fringe(m, n)
    8

    **References**

    [1] https://en.wikipedia.org/wiki/Zernike_polynomials
    """
    if m >= 0:
        j = (1 + (n + m) // 2) ** 2 - 2 * m
    else:
        j = (1 + (n - m) // 2) ** 2 + 2 * m + 1
    return j


def _to_mn(index, convention):
    """Helper function to converts given index (single) to mn indices (tuple
    of ints)

    :param index: index to be converted to m, n
    :type index: int or Tuple(int, int)
    :param convention: used indexing convention
    :type convention: str
    :return: m-angular frequency, n-radial order
    :rtype: Tuple(int, int)
    """
    _convention = convention.lower()
    if _convention == 'mn':
        m, n = index
    elif _convention == 'noll':
        m, n = noll_to_mn(index)
    elif _convention == 'osa':
        m, n = osa_to_mn(index)
    elif _convention == 'fringe':
        m, n = fringe_to_mn(index)
    else:
        raise ValueError(
            f'Zernike convention `{convention}` is not recognized! Only `mn`,'
            f' `osa`, `noll`, and `fringe` are available.'
        )
    return m, n


def construct_wavefront(
        index_coefs_dict, convention, rho, theta, normalize=True, fill_value=0
):
    """Sums up zernikes with given coefs to form the final wavefront:

    wf = sum(coef * zernike(index, convention, rho, theta) for index, coef in index_coefs_dict.items())

    :param index_coefs_dict: a dict with corresponding coefficients
    :type index_coefs_dict: dict
    :param convention: used indexing convention
    :type convention: str
    :param rho: r coordinates of the plane,
    :type rho: numpy.array
    :param theta: calculated radial function
    :type theta: numpy.array
    :param normalize: should wavefront be normalized?
    :type normalize: bool
    :param fill_value: what value should be put for r > 1
    :type fill_value: numeric
    :return: fitted wf (summed zernikes)
    :rtype: numpy.array

    **Example**

    >>> import astropy.units as u
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> import pyoptica as po
    >>> # Now we construct array which will be used for fitting
    >>> npix = 101
    >>> pixel_scale = 0.5 * u.m
    >>> x, y = po.utils.mesh_grid(npix, pixel_scale)
    >>> r_max = .5 * npix * pixel_scale
    >>> r, theta = po.utils.cart2pol(x, y)
    >>> r = r / r_max
    >>> r, theta = r.value, theta.value
    >>> # Now it is time to construct a wf (sum up zernikes):
    >>> coefs = [0.1] * 12
    >>> js = list(range(12))
    >>> index_coefs_dict = dict(zip(js, coefs))
    >>> convention = 'noll'
    >>> wf = po.zernikes.construct_wavefront(index_coefs_dict, convention, r, theta)

    """
    wf = np.zeros(rho.shape)
    for index, w in index_coefs_dict.items():
        wf += w * zernike(
            index, convention, rho, theta, normalize, fill_value=fill_value)
    return wf


def fit_zernikes(
        wf, indices, convention, rho, theta, normalize=True, cache=True,
        method='COBYLA', **kwargs):
    """Fits zernikes to the given wf -- for given `indices` finds corresponding
    coefficients `c_j` that when summed form fitted_wf:
        fitted_wf = sum((c_j * zernike_j(j, rho, theta, normalize))
    for which the difference (more specifically: l2-norm of the difference) is
    minimized.

    For the minimization routine `scipy.optimize.minimize` is used. For details
    please go to: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html

    :param wf: a wavefront to be fitted
    :type wf: numpy.array
    :param indices: a list of zernikes to be fitted
    :type indices: iterable
    :param convention: used indexing convention
    :type convention: str
    :param rho: r coordinates of the plane,
    :type rho: numpy.array
    :param theta: calculated radial function
    :type theta: numpy.array
    :param normalize: should wavefront be normalized?
    :type normalize: bool
    :param cache: should zernikes be cached?
    :type cache: bool
    :param method: method of `scipy.optimize.minimize`
    :type method: str
    :param kwargs: remaining arguments for `scipy.optimize.minimize`
    :return: (found coefficients, minimization results)
    :rtype: Tuple(dict, OptimizeResult)

    **Example**

    >>> import astropy.units as u
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> # And from pyoptica
    >>> import pyoptica as po
    >>> # Now we construct array which will be used for fitting
    >>> npix = 101
    >>> pixel_scale = 0.5 * u.m
    >>> x, y = po.utils.mesh_grid(npix, pixel_scale)
    >>> r_max = .5 * npix * pixel_scale
    >>> r, theta = po.utils.cart2pol(x, y)
    >>> r = r / r_max
    >>> r, theta = r.value, theta.value
    >>> # Now it is time to construct a wf (sum up zernikes):
    >>> coefs = [0.1] * 12
    >>> indices = list(range(12))
    >>> js_coefs_dict = dict(zip(indices, coefs))
    >>> convention = 'noll'
    >>> wf = po.zernikes.construct_wavefront(js_coefs_dict, convention, r, theta)
    >>> fit, res = po.zernikes.fit_zernikes(wf, indices, convention, r, theta, cache=True)
    >>> fit
        {0: 0.09999999466400215,
         1: 0.09999998903721308,
         2: 0.09999999890069368,
         3: 0.09999999483945994,
         4: 0.09999999212142557,
         5: 0.0999999933147726,
         6: 0.0999999933135079,
         7: 0.09999999519388976,
         8: 0.09999999575002586,
         9: 0.0999999900420963,
         10: 0.09999999224797648,
         11: 0.0999999967091342}

    """
    if np.isnan(wf).any():
        raise ValueError(
            'Nan values are not accepted in the fitting routine. Please use'
            ' a different `fill_value` while constructing input wf!'
        )
    initial_coefs = np.zeros(len(indices))
    if cache:
        cached_zernikes = [
            zernike(i, convention, rho, theta, normalize) for i in indices]
        res = minimize(
            _l2_norm_of_aberration_wavefront_cache,
            x0=initial_coefs,
            args=(wf, cached_zernikes),
            method=method,
            **kwargs)
    else:
        res = minimize(
            _l2_norm_of_aberration_wavefront,
            x0=initial_coefs,
            args=(indices, convention, wf, rho, theta, normalize),
            method=method,
            **kwargs)
    return dict(zip(indices, res.x)), res


def fit_zernikes_lstsq(wf, indices, convention, rho, theta, normalize=True):
    """ Fits zernikes using the least squares method:

            Zi * ci = wf
    Where Zi and ci represent vectors with zernikes and coefficients
    respectively.

    This is the recommended method to fit a wavefront.

    :param wf: a wavefront to be fitted
    :type wf: numpy.array
    :param js: a list of zernikes to be fitted
    :type js: iterable
    :param convention: used indexing convention
    :type convention: str
    :param rho: r coordinates of the plane,
    :type rho: numpy.array
    :param theta: calculated radial function
    :type theta: numpy.array
    :param normalize: should wavefront be normalized?
    :type normalize: bool
    :return: fitted coeffs in a dict, (residuals, rank, singular_values)
    :rtype: Tuple(dict, tuple)

    **Example**

    >>> import astropy.units as u
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> # And from pyoptica
    >>> import pyoptica as po
    >>> # Now we construct array which will be used for fitting
    >>> npix = 101
    >>> pixel_scale = 0.5 * u.m
    >>> x, y = po.utils.mesh_grid(npix, pixel_scale)
    >>> r_max = .5 * npix * pixel_scale
    >>> r, theta = po.utils.cart2pol(x, y)
    >>> r = r / r_max
    >>> r, theta = r.value, theta.value
    >>> # Now it is time to construct a wf (sum up zernikes):
    >>> coefs = [0.1] * 12
    >>> indices = list(range(12))
    >>> js_coefs_dict = dict(zip(indices, coefs))
    >>> convention = 'noll'
    >>> wf = po.zernikes.construct_wavefront(js_coefs_dict, convention, r, theta)
    >>> fit, res = po.zernikes.fit_zernikes_lstsq(wf, indices, r, theta)
    >>> fit
        {0: 0.10000000000000002,
         1: 0.10000000000000016,
         2: 0.10000000000000019,
         3: 0.1,
         4: 0.09999999999999987,
         5: 0.10000000000000009,
         6: 0.09999999999999998,
         7: 0.10000000000000002,
         8: 0.10000000000000009,
         9: 0.10000000000000003,
         10: 0.1,
         11: 0.10000000000000003}

    """

    # It is not possible to work on 2D arrays... Therefore the need to be
    # flattened. That doesn't matter though -- it is just a way of data
    # representation.
    zernikes = np.array(
        [zernike(i, convention, rho, theta, normalize).flatten() for i in
         indices]
    ).T
    coefs, *res = np.linalg.lstsq(zernikes, wf.flatten(), rcond=None)
    return dict(zip(indices, coefs)), res


def _l2_norm_of_aberration_wavefront_cache(coefs, wf, cached_zernikes):
    """Calculates l2-norm between wavefront and a sum of zernikes in
    `cached_zernikes` and corresponding coefficients `coefs`.

    To be used in an optimization routine.

    :param coefs: a list of coefficients corresponding to zernikes
    :type coefs: iterable
    :param wf: a wavefront with which l2-norm is calculated
    :type wf: numpy.array
    :param cached_zernikes: a list of precalculated zernikes
    :type cached_zernikes: iterable
    :return: l2-norm of the difference between the two wavefronts
    :type: float

    """
    constructed_wf = np.zeros_like(wf)
    for z_dist, w in zip(cached_zernikes, coefs):
        constructed_wf += w * z_dist
    norm = np.linalg.norm(wf - constructed_wf)
    logger.debug(f"L2-norm = {norm}")
    return norm


def _l2_norm_of_aberration_wavefront(
        coefs, indices, convention, wavefront, rho, theta, normalize):
    """
    Calculates l2-norm between wavefront and a sum of zernikes of indices `js`
    and corresponding coefficients `coefs`.

    To be used in an optimization routine.

    :param coefs: a list of coefficients corresponding to zernikes
    :type coefs: iterable
    :param indices: a list of zernikes
    :type indices: iterable
    :param convention: used indexing convention
    :type convention: str
    :param wavefront: a wavefront with which l2-norm is calculated
    :type wavefront: numpy.array
    :param rho: r coordinates of the plane,
    :type rho: numpy.array
    :param theta: calculated radial function
    :type theta: numpy.array
    :param normalize: should wavefront be normalized?
    :type normalize: bool
    :return: l2-norm of the difference between the two wavefronts
    :type: float

    """
    index_coef = dict(zip(indices, coefs))
    sum_wf = construct_wavefront(index_coef, convention, rho, theta, normalize)
    norm = np.linalg.norm(wavefront - sum_wf)
    logger.debug(f"L2-norm = {norm}")
    return norm


def get_zernike_name(index, convention, latex=True):
    """Gets name of the zernike for given convention for index < 17 (Fringe)
    returns also the name.

    :param index: index of zernike in given convention
    :type index: int or Tuple(int, int)
    :param convention: used indexing convention
    :type convention: str
    :param latex: add latex equation formatting?
    :type latex: bool
    :return: name of zernike
    :rtype: str

    **Example**

    >>> import pyoptica as po
    >>> po.zernikes.get_zernike_name(7, 'fringe')
    '$Z_{7}$ x-Coma'
    >>> po.zernikes.get_zernike_name(7, 'fringe', latex=False)
    'Z_7 x-Coma'
    """
    _convention = convention.lower()
    m, n = _to_mn(index, convention)
    if _convention in ['noll', 'osa', 'fringe']:
        if latex:
            name = f'$Z_{{{index}}}$'
        else:
            name = f'Z_{index}'
    else:
        if latex:
            name = f'$Z_{{{n}}}^{{{m}}}$'
        else:
            name = f'Z_{n}^{m}'

    if (m, n) in ZERNIKE_NAMES:
        name += ' ' + ZERNIKE_NAMES[(m, n)]
    return name
