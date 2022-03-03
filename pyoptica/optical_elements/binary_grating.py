import astropy.units as u
import numpy as np
from scipy.ndimage import rotate

from .base_optical_element import BaseOpticalElement
from .. import utils


class BinaryGrating(BaseOpticalElement):
    """ A class representing a binary amplitude/phase grating.

    :param period: Period of the grating.
    :type period: astropy.Quantity of type length
    :param duty: Duty cycle of the grating [0-1].
    :type duty: float
    :param angle: Angle of the grating in degrees.
    :type angle: astropy.Quantity of type angle
    :param kind: Type of grating ["amplitude", "phase"].
    :type kind: str
    :param a_trans: Transmission of the grating [0-1].
    :type a_trans: float
    :param o_thick: Optical thickness of the grating.
    :type o_thick: float


    **Example**

    >>> import astropy.units as u
    >>> import pyoptica as po
    >>>
    >>> period = 1 * u.mm
    >>> duty = 0.5
    >>> angle = 45 * u.deg
    >>> grating = po.BinaryGrating(period, duty, angle)

    """

    VALID_RANGE_DUTY = (0, 1)
    VALID_RANGE_ANGLE = (0 * u.deg, 360 * u.deg)
    VALID_RANGE_A_TRANS = (0, 1)
    VALID_RANGE_O_THICK = (0, 2 * np.pi)
    VALID_KIND = ["amplitude", "phase"]

    def __init__(self, period, duty=.5, angle=0 * u.deg, kind='amplitude', a_trans=1, o_thick=np.pi):
        self.period = period
        self.duty = utils.range_validator(duty, self.VALID_RANGE_DUTY)
        self.angle = utils.range_validator(angle, self.VALID_RANGE_ANGLE)
        self.kind = utils.param_validator(kind, self.VALID_KIND)
        self.a_trans = utils.range_validator(a_trans, self.VALID_RANGE_A_TRANS)
        self.o_thick = utils.range_validator(o_thick, self.VALID_RANGE_O_THICK)

    @classmethod
    def etch_grating(cls, npix, period, duty, angle=0):

        def _crop_center(img, cropx, cropy):
            y, x = img.shape
            startx = x // 2 - (cropx // 2)
            starty = y // 2 - (cropy // 2)
            return img[starty:starty + cropy, startx:startx + cropx]

        cell = [1] * int(period * duty) + [0] * int(period * (1 - duty))
        if angle:
            pad = np.sqrt(2)
        else:
            pad = 1
        profile_r = np.hstack([cell] * (int(pad * npix / period * .5 + period)))[:int(npix * pad / 2)]
        profile_l = np.hstack([cell[::-1]] * (int(pad * npix / period * .5 + period)))[:int(npix * pad / 2)][::-1]
        profile = np.hstack([profile_l, profile_r])
        grating = np.vstack([profile] * int(npix * pad))
        if angle:
            return _crop_center(rotate(grating, angle=angle, reshape=True), npix, npix)
        else:
            return grating

    def amplitude_transmittance(self, wavefront):
        if "amplitude" in self.kind:
            npix = wavefront.npix
            pix_period = (self.period.to(u.um).value // wavefront.pixel_scale.to(u.um).value)
            self.logger.info(f"Resulting period due to canvas granularity: "
                             f"{(pix_period * wavefront.pixel_scale).to(u.um)}"
                             )
            gr = self.etch_grating(npix, pix_period, self.duty, self.angle.to(u.deg).value) * self.a_trans
            return gr
        else:
            return np.ones_like(wavefront.amplitude)

    def phase_transmittance(self, wavefront):
        if "phase" in self.kind:
            npix = wavefront.npix
            pix_period = (self.period.to(u.um).value // wavefront.pixel_scale.to(u.um).value)
            self.logger.info(f"Resulting period due to canvas granularity: "
                             f"{(pix_period * wavefront.pixel_scale).to(u.um)}"
                             )
            gr = self.etch_grating(npix, pix_period, self.duty, self.angle.to(u.deg).value) * self.a_trans
            return np.exp(1.j * self.o_thick * gr)
        else:
            return np.exp(1.j * np.zeros_like(wavefront.phase))
