import abc

from .. import logging
from ..wavefront import Wavefront


class BaseOpticalElement(logging.LoggerMixin, abc.ABC):
    """ An abstract class for optical elements. It is the base class for all
    future optical elements (e.g. a lens). All classes derived from this base
    one should implement `amplitude_transmittance` and `phase_transmittance`.
    They describe respectively how the element modulates amplitude and how it
    shifts phase of the wavefront the element interacts with.

    """

    @abc.abstractmethod
    def amplitude_transmittance(self, wavefront):
        pass

    @abc.abstractmethod
    def phase_transmittance(self, wavefront):
        pass

    def transmittance(self, wavefront):
        """Produces transmittance of the optical element"""
        t_amplitude = self.amplitude_transmittance(wavefront)
        t_phase = self.phase_transmittance(wavefront)
        t = t_amplitude * t_phase
        return t

    def __mul__(self, wavefront):
        """Multiplication with a wavefront.

        :param wavefront: Main pyoptica `Wavefront` class.
        :type wavefront: pyoptica.Wavefront
        :return: multiplied wavefront.
        :rtype: pyoptica.Wavefront
        """

        multiplied_wf = Wavefront(wavefront.wavelength, wavefront.pixel_scale, wavefront.npix)
        multiplied_wf.wavefront = wavefront.wavefront * self.transmittance(wavefront)
        return multiplied_wf

    __rmul__ = __mul__
