import astropy.units as u
import numpy as np

from .base_optical_element import BaseOpticalElement


class RectangularAperture(BaseOpticalElement):
    """ A class representing a rectangular aperture in an optical system.

    :param width: width of the aperture
    :type width: astropy.Quantity of type length
    :param height: heigth of the aperture
    :type height: astropy.Quantity of type length

    **Example**

    >>> import astropy.units as u
    >>> import pyoptica as po
    >>>
    >>> width = 1 * u.mm
    >>> height = 1 * u.mm
    >>> aperture = po.RectangularAperture(width, height)

    **References**

    [1] Joseph W. Goodman (2004) -
    *Introduction to Fourier Optics*, W. H. Freeman

    """

    @u.quantity_input(width=u.m, height=u.m)
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def amplitude_transmittance(self, wavefront):
        """Amplitude is fully transmitted for dimensions smaller than aperture size.

        :param wavefront: The wavefront which interacts with the aperture
        :type wavefront: pyoptica.Wavefront
        :return: distribution of amplitude transmittance resulting from the aperture
        :rtype: numpy.array

        """
        x, y = wavefront.x, wavefront.y
        return np.where((np.abs(x) <= (.5 * self.width)), 1, 0) * \
               np.where(np.abs(y) <= (.5 * self.height), 1, 0)

    def phase_transmittance(self, wavefront):
        r"""Phase is not disturbed for dimensions smaller than aperture radius.

        :param wavefront: The wavefront which interacts with the aperture
        :type wavefront: pyoptica.Wavefront
        :return: distribution of phase transmittance resulting from the aperture
        :rtype: numpy.array

        """

        return np.exp(1.j * np.zeros_like(wavefront.phase))


class RectangularObscuration(RectangularAperture):
    """ A class representing a rectangular obscuration in an optical system.

    :param width: width of the aperture
    :type width: astropy.Quantity of type length
    :param height: heigth of the aperture
    :type height: astropy.Quantity of type length

    **Example**

    >>> import astropy.units as u
    >>> import pyoptica as po
    >>>
    >>> width = 1 * u.mm
    >>> height = 1 * u.mm
    >>> obscuration = po.RectangularObscuration(width, height)

    **References**

    [1] Joseph W. Goodman (2004) -
    *Introduction to Fourier Optics*, W. H. Freeman

    """

    def amplitude_transmittance(self, wavefront):
        return 1.0 - super().amplitude_transmittance(wavefront)
