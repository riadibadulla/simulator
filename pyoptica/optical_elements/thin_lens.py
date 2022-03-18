import warnings

import astropy.units as u
import numpy as np
import torch
from .base_optical_element import BaseOpticalElement
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class ThinLens(BaseOpticalElement):
    """ A class representing a thin lens. The reasoning used to develop
    functionality of the lens is based on chapter 5 of [1].

    :param radius: radius of the lens
    :type radius: astropy.Quantity of type length
    :param f: focal distance of the lens
    :type f: astropy.Quantity of type length

    **Example**

    >>> import astropy.units as u
    >>> import pyoptica as po
    >>>
    >>> f = 1 * u.cm
    >>> radius = 1 * u.mm
    >>> lens = po.ThinLens(radius, f)

    **References**

    [1] Joseph W. Goodman (2004) -
    *Introduction to Fourier Optics*, W. H. Freeman

    """

    @u.quantity_input(radius=u.m, f=u.m)
    def __init__(self, radius, f, wavefront):
        self.radius = radius
        self.f = f
        self.phase_transmittance_precalculated = self.set_phase_transmittance(wavefront)

    def amplitude_transmittance(self, wavefront):
        """In the thin lens approximation amplitude is fully transmitted.

        :param wavefront: The wavefront which interacts with the lens
        :type wavefront: pyoptica.Wavefront
        :return: distribution of amplitude transmittance resulting from the lens
        :rtype: numpy.array

        """
        return torch.ones_like(wavefront.amplitude).to(device)

    def set_phase_transmittance(self, wavefront):
        r"""Calculations of phase transmittance based on eq. 5.10 in [1]:

                :math:`t_{l}(x, y)=\exp \left[-j \frac{k}{2 f}\left(x^{2}+y^{2}\right)\right]`

                :param wavefront: The wavefront which interacts with the lens
                :type wavefront: pyoptica.Wavefront
                :return: distribution of phase transmittance resulting from the lens
                :rtype: numpy.array

                """
        # self._check_sampling(wavefront)
        k = wavefront.k
        x, y = wavefront.x, wavefront.y
        xy_squared = x ** 2 + y ** 2
        t1 = np.exp(-(1.j * k) / (2 * self.f) * xy_squared)
        phi = np.where(
            xy_squared <= self.radius ** 2, t1, 1
        )
        # TODO: maybe need to tensor entire function
        return torch.tensor(phi).to(device)

    def phase_transmittance(self, wavefront):
        return self.phase_transmittance_precalculated

    def _check_sampling(self, wavefront):
        """Checks if sampling of the given wavefront meets the requirement:

        :matha: `\Delta x  \leq \frac{|f|}{D_{L}} \lambda`

        Based on eq. 6.14 in  David Voelz (2011)
        *Computational Fourier Optics Matlab Tutorial*, Spie Press

        :param wavefront: wavefront to be multiplied by the lens
        :type wavefront: pyoptica.Wavefront

        """
        min_sampling = np.abs(self.f) / (
                    2 * self.radius) * wavefront.wavelength
        # Just for readable output.
        min_sampling = min_sampling.to(wavefront.pixel_scale.unit)
        # if wavefront.pixel_scale > min_sampling:
        #     warning = f'Bad sampling: {wavefront.pixel_scale} > {min_sampling}'
        #     self.logger.warning(warning)
        #     warnings.warn(warning)
