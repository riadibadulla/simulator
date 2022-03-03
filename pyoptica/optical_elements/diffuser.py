import numpy as np

from .base_optical_element import BaseOpticalElement


class Diffuser(BaseOpticalElement):
    """ A class representing an optical diffuser.

    :param distribution: Distribution of phase shift ['uniform', 'normal'].
    :type distribution: str
    :param rotating: If *True* phase changes from call to call.
    :type rotating: bool

    **Example**

    >>> import astropy.units as u
    >>> import pyoptica as po
    >>>
    >>> distribution = 'normal'
    >>> rotating = True
    >>> diff = po.Diffuser(distribution)

    **References**

    [1] Joseph W. Goodman (2004) -
    *Introduction to Fourier Optics*, W. H. Freeman

    """
    DISTRIBUTIONS = ['uniform', 'normal']

    def __init__(self, distribution='uniform', rotating=False):

        if distribution not in self.DISTRIBUTIONS:
            raise ValueError(f"Valid distributions are: {self.DISTRIBUTIONS}")

        self.distribution = distribution
        self.rotating = rotating
        self._rnd_state = None
        if distribution == 'uniform':
            self._distribution_generator = np.random.rand
        elif distribution == 'normal':
            self._distribution_generator = np.random.randn
        self._seed = np.random.randint(0, 2**16-1)

    def _reset_seed(self):
        if not self.rotating:
            self._rnd_state = np.random.RandomState(self._seed)
            if self.distribution == 'uniform':
                self._distribution_generator = self._rnd_state.rand
            if self.distribution == 'normal':
                self._distribution_generator = self._rnd_state.randn

    def amplitude_transmittance(self, wavefront):
        """
        We assumed that the diffuser's amplitude transmittance is one .

        :param wavefront: The wavefront which interacts with the diffuser
        :type wavefront: pyoptica.Wavefront
        :return: distribution of amplitude transmittance resulting from the diffuser
        :rtype: numpy.array
        """

        return np.ones_like(wavefront.amplitude)

    def phase_transmittance(self, wavefront):
        """
        We assumed that the diffuser introduces completely random phase.

        :param wavefront: The wavefront which interacts with the diffuser
        :type wavefront: pyoptica.Wavefront
        :return: distribution of amplitude transmittance resulting from the diffuser
        :rtype: numpy.array
        """
        if not self.rotating:
            self._reset_seed()
        return np.exp(1.j * 2 * np.pi * (self._distribution_generator(*wavefront.phase.shape) - .5))
