import astropy.units as u
import numpy as np

from .base_optical_element import BaseOpticalElement
from .. import utils
from ..wavefront import Wavefront
import torch

class FreeSpace(BaseOpticalElement):
    """ A class representing an free space for propagation.

    :param distance: Length of the free space to propagate.
    :type distance: astropy.Quantity of type length.
    :param method: Method of prpagation calculation.
    :type method: str

    **Example**

    >>> import astropy.units as u
    >>> import pyoptica as po
    >>>
    >>> fs = po.FreeSpace(10 * u.cm, 'ASPW')

    **References**

    [1] Joseph W. Goodman (2004) -
    *Introduction to Fourier Optics*, W. H. Freeman

    """
    METHODS = ['ASPW']  # , 'Fresnel', 'Fraunhofer']

    def __init__(self, distance, method='ASPW'):

        if method not in self.METHODS:
            raise ValueError(f"Valid methods are: {self.METHODS}")

        self.distance = distance
        self.method = method

    def __mul__(self, wavefront):
        return self.propagate(wavefront)

    __rmul__ = __mul__

    @classmethod
    def calc_optimal_distance(cls, wavefront):
        r"""Based on [2], this function calulates the optimal distance of
        propagation:
        :math:`z = \frac{N\left( \Delta x \right)^2}{\lambda}`
        for given `wavefront`.

        :return: The optimal distance of propagate
        :rtype: astropy.Quantity of type length

        >>> import astropy.units as u
        >>> import pyoptica as po
        >>>
        >>> wavelength = 633 * u.nm
        >>> pixel_scale = 100 * u.nm
        >>> npix = 2048
        >>> distance = 60 * u.um
        >>> wf = po.Wavefront(wavelength, pixel_scale, npix)
        >>> fs = po.FreeSpace(distance, "ASPW")
        >>> fs.calc_optimal_distance(wf)
        <Quantity 32.35387 um>

        """

        pixel_scale = wavefront.pixel_scale
        wavelength = wavefront.wavelength
        npix = wavefront.npix
        z = (pixel_scale ** 2 * npix / wavelength).decompose()
        return z

    @u.quantity_input(distance=u.m)
    def calc_propagation_steps(self, wavefront):
        r"""For the given `wavefront` it calculates number of steps and their size
        required to propagate to the given `distance` based on the sampling
        requirements:

        :math:`z_{sampling} = \frac{N\left( \Delta x \right)^2}{\lambda}`

        :math:`n = \left \lceil  \frac{z}{z_{sampling}} \right \rceil`

        :math:`z_{size} = \frac{z}{n}`

        :param wavefront: wavefront propagating through free space.
        :type wavefront: pyoptica.Wavefront
        :return: number of required steps and their size
        :rtype: Tuple(int, astropy.Qunatity of type length)

        >>> import astropy.units as u
        >>> import pyoptica as po
        >>>
        >>> wavelength = 633 * u.nm
        >>> pixel_scale = 100 * u.nm
        >>> npix = 2048
        >>> distance = 60 * u.um
        >>> wf = po.Wavefront(wavelength, pixel_scale, npix)
        >>> fs = po.FreeSpace(distance)
        >>> fs.calc_propagation_steps(wf)
        (2, <Quantity 30 um>)

        """
        abs_distance = np.abs(self.distance)
        optimal_distance = self.calc_optimal_distance(wavefront)
        no_steps = np.ceil((abs_distance / optimal_distance).decompose()).value
        no_steps = int(no_steps)
        step_size = self.distance / no_steps
        return no_steps, step_size

    def propagate(self, wavefront):
        r"""A method for propagate using Angular Spectrum of Plane Waves. It is
        relatively straightforward to compute; following [1] (Chapter 5.1):

        :math:`U_{2}(x, y)=\mathfrak{F}^{-1}\left\{\mathfrak{F}
        \left\{U_{1}(x, y)\right\} H\left(f_{X}, f_{Y}\right)\right\}`

        where:

        :math:`U_{1}(x, y)` -- the initial wavefront,

        :math:`H\left(f_{X}, f_{Y}\right)=e^{j k z} \exp \left[-j \pi
        \lambda z\left(f_{X}^{2}+f_{Y}^{2}\right)\right]`
        -- the transfer fuction.

        It is computed using the following convolution theorem:

        :math:`f * g =\mathfrak{F}^{-1}\left\{\mathfrak{F}
        \{ f \}\mathfrak{F} \{g\} \right\}`

        Important note: the distance of propagate is limited by sampling [2]:
        :math:`z \leq \frac{N\left( \Delta x \right)^2}{\lambda}`
        If the given `distance` to propagate is greater than the limit given by
        sampling the propagate will be done in multiple equidistant steps
        (summing to the given `distance`).

        :param wavefront: Wavefront to propaget
        :type wavefront: pyoptica.Wavefront
        :return: Propagated wavefront
        :rtype: pyoptica.Wavefront

        **References**

        [1] David Voelz (2011) -
        *Computational Fourier Optics Matlab Tutorial*, Spie Press

        [2] David Voelz and Michael C. Roggemann (2009) -
        *Digital simulation of scalar optical diffraction: revisiting chirp
        function sampling criteria and consequences*, Applied Optics

        >>> import astropy.units as u
        >>> import pyoptica as po
        >>>
        >>> wavelength = 633 * u.nm
        >>> pixel_scale = 100 * u.nm
        >>> npix = 2048
        >>> distance = 60 * u.um
        >>> wf = po.Wavefront(wavelength, pixel_scale, npix)
        >>> fs = po.FreeSpace(distance)
        >>> wf_at_distance = wf * fs

        """
        propagated_wf = Wavefront(wavefront.wavelength, wavefront.pixel_scale, wavefront.npix)
        steps_number, step_size = self.calc_propagation_steps(wavefront)
        h = FreeSpace(step_size, method=self.method).phase_transmittance(wavefront)
        H = utils.fft(h)
        wf_at_distance = utils.fft(wavefront.wavefront)
        for _ in range(steps_number):
            wf_at_distance = wf_at_distance * H
        wf_at_distance = utils.ifft(wf_at_distance)
        propagated_wf.wavefront = wf_at_distance
        # self.logger.info(
        #     f"Propagated using '{self.method}' to {self.distance:.0}.")
        return propagated_wf

    def amplitude_transmittance(self, wavefront):
        """We assumed that the free space amplitude transmittance is one .

        :param wavefront: The wavefront which interacts with the free space.
        :type wavefront: pyoptica.Wavefront
        :return: distribution of amplitude transmittance resulting from the free space.
        :rtype: numpy.array

        """
        return torch.ones_like(wavefront.amplitude)

    def phase_transmittance(self, wavefront):
        """We assumed that the perfect diffuser introduces completely random phase.

        :param wavefront: The wavefront which interacts with the free space.
        :type wavefront: pyoptica.Wavefront
        :return: distribution of phase transmittance resulting from the free space.
        :rtype: numpy.array
        """
        H = torch.zeros_like(wavefront.wavefront)
        pix_scale_m = wavefront.pixel_scale.to(u.m).value
        wavelen_m = wavefront.wavelength.to(u.m).value
        distance_m = self.distance.to(u.m).value

        x = wavefront.x.to(u.m).value
        y = wavefront.y.to(u.m).value
        npix = wavefront.npix
        f_x = (x / (pix_scale_m ** 2 * npix))
        f_y = (y / (pix_scale_m ** 2 * npix))
        k = wavefront.k.to(1 / u.m).value

        if self.method == "ASPW":
            rhosqr = f_x ** 2 + f_y ** 2
            exp_1 = 1.j * k * distance_m
            exp_2 = -1.j * np.pi * wavelen_m * distance_m * rhosqr
            # H = np.exp(exp_1 + exp_2)
            #TODO: may need to edit ks and xys to be tensor from the begining
            H = torch.exp(torch.tensor(exp_1) + torch.tensor(exp_2))
        return utils.ifft(H)
