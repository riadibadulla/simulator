import astropy.units as u
import numpy as np

from .base_optical_element import BaseOpticalElement


class CircularAperture(BaseOpticalElement):
    """ A class representing a circular diaphragm in an optical system.

    :param radius: radius of the aperture
    :type radius: astropy.Quantity of type length

    **Example**

    >>> import astropy.units as u
    >>> import pyoptica as po
    >>>
    >>> radius = 1 * u.mm
    >>> aperture = po.CircularAperture(radius)

    **References**

    [1] Joseph W. Goodman (2004) -
    *Introduction to Fourier Optics*, W. H. Freeman

    """

    @u.quantity_input(radius=u.m)
    def __init__(self, radius):
        self.radius = radius

    def amplitude_transmittance(self, wavefront):
        """Amplitude is fully transmitted for radii smaller than aperture radius.

        :param wavefront: The wavefront which interacts with the aperture
        :type wavefront: pyoptica.Wavefront
        :return: distribution of amplitude transmittance resulting from the aperture
        :rtype: numpy.array

        """
        x, y = wavefront.x, wavefront.y
        xy_squared = x ** 2 + y ** 2
        return np.where(xy_squared <= self.radius ** 2, 1, 0)

    def phase_transmittance(self, wavefront):
        r"""Phase is not disturbed for radii smaller than aperture radius.

        :param wavefront: The wavefront which interacts with the aperture
        :type wavefront: pyoptica.Wavefront
        :return: distribution of phase transmittance resulting from the aperture
        :rtype: numpy.array

        """

        return np.exp(1.j * np.zeros_like(wavefront.phase))


class CircularObscuration(CircularAperture):
    """ A class representing a circular obscuration in an optical system.

       :param radius: radius of the aperture
       :type radius: astropy.Quantity of type length

       **Example**

       >>> import astropy.units as u
       >>> import pyoptica as po
       >>>
       >>> radius = 1 * u.mm
       >>> obscuration = po.CircularObscuration(radius)

       **References**

       [1] Joseph W. Goodman (2004) -
       *Introduction to Fourier Optics*, W. H. Freeman

       """

    def amplitude_transmittance(self, wavefront):
        return 1.0 - super().amplitude_transmittance(wavefront)
