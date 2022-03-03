import numpy as np

from .base_optical_element import BaseOpticalElement
from .. import utils


class SiemensStar(BaseOpticalElement):
    """ A class representing a binary amplitude/phase Siemens Star.

    :param cycles: Number of Siemens Star cycles.
    :type cycles: int
    :param kind: Type of Siemens Star ["amplitude", "phase"].
    :type kind: str
    :param a_trans: Transmission of the Siemens Star [0-1].
    :type a_trans: float
    :param o_thick: Optical thickness of the Siemens Star.
    :type o_thick: float


    **Example**

    >>> import astropy.units as u
    >>> import pyoptica as po
    >>>
    >>> cycles = 5
    >>> star = po.SiemensStar(cycles)

    """

    VALID_RANGE_A_TRANS = (0, 1)
    VALID_RANGE_O_THICK = (0, 2 * np.pi)
    VALID_KIND = ["amplitude", "phase"]

    def __init__(self, cycles, kind='amplitude', a_trans=1, o_thick=np.pi):
        self.cycles = cycles
        self.kind = utils.param_validator(kind, self.VALID_KIND)
        self.a_trans = utils.range_validator(a_trans, self.VALID_RANGE_A_TRANS)
        self.o_thick = utils.range_validator(o_thick, self.VALID_RANGE_O_THICK)

    def amplitude_transmittance(self, wavefront):
        if "amplitude" in self.kind:
            _, phi = utils.cart2pol(wavefront.x, wavefront.y)
            return np.where(np.sin(self.cycles * phi) >= 0, 1, 0)
        else:
            return np.ones_like(wavefront.amplitude)

    def phase_transmittance(self, wavefront):
        if "phase" in self.kind:
            _, phi = utils.cart2pol(wavefront.x, wavefront.y)
            ss = np.where(np.sin(self.cycles * phi) >= 0, 1, 0)
            return np.exp(1.j * self.o_thick * ss)
        else:
            return np.exp(1.j * np.zeros_like(wavefront.phase))
