import astropy.units as u
import numpy as np
from astropy.io import fits

from . import distribution


class Wavefront(distribution.Distribution):
    """A class representing a wavefront.

    :param wavelength: Wavelength of simulated wavefront
    :type wavelength: astropy.Quantity of type length
    :param pixel_scale: Sampling of the wavefront
    :type pixel_scale: astropy.Quantity of type length
    :param npix: Number of pixels in the wavefront array
    :type npix: int

    :raises TypeError: wavelength and pixelscale are expected to be of type
        astropy.quantity of type length

    **Example**

    >>> import astropy.units as u
    >>> import pyoptica as po
    >>>
    >>> wavelength = 500 * u.nm
    >>> pixel_scale = 22 * u.um
    >>> npix = 1024
    >>> wf = po.Wavefront(wavelength, pixel_scale, npix)

    **References**

    [1] Joseph W. Goodman (2004) - *Introduction to Fourier Optics*, W. H. Freeman

    [2] Kedar Khare (2016) - *Fourier Optics and Computational Imaging*, Wiley&Sons Ltd.

    [3] David Voelz (2011) - *Computational Fourier Optics Matlab Tutorial*, Spie Press

    """
    PLOTTING_OPTIONS = {
        'intensity': dict(x='x', y='y', title='Intensity'),
        'amplitude': dict(x='x', y='y', title='Amplitude'),
        'phase': dict(
            colorbar_title='[rad]', vmin=-np.pi, vmax=np.pi,
            bar_ticks=[-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
            bar_ticks_labels=[
                r'$-\pi$',
                r'$-\frac{\pi}{2}$',
                r'$0$',
                r'$\frac{\pi}{2}$',
                r'$\pi$'
            ],
            x='x', y='y', cmap='hsv', title='Phase',
            z_function='_apply_phase_plotting_threshold'
        )
    }
    @u.quantity_input(wavelength=u.m, pixel_scale=u.m)
    def __init__(self, wavelength, pixel_scale, npix):
        super().__init__(wavelength, pixel_scale, npix)
        self.wavefront = np.ones((npix, npix), dtype=np.complex64)
        self.logger.debug(f"Created {self}")

    @property
    def amplitude(self):
        """Get/Set amplitude of the wavefront. Taking/returning
        an array of shape=(npix, npix).

        :getter: Amplitude of the wavefront.
        :setter: Loads given array as amplitude of the wavefront,
            leaving phase unchanged.
        :type: np.array
        """
        return np.abs(self.wavefront)

    @amplitude.setter
    def amplitude(self, amplitude):
        self.wavefront = amplitude * np.exp(1.j * self.phase)

    @property
    def intensity(self):
        """Get/Set Intensity of the wavefront. Taking/returning
        an array of shape=(npix, npix) .

        :getter: Intensity of the wavefront.
        :setter: Loads given array as intensity of the wavefront,
            leaving phase unchanged.
        :type: np.array

        """
        return self.amplitude ** 2

    @intensity.setter
    def intensity(self, intensity):
        self.wavefront = np.sqrt(intensity) * np.exp(1.j * self.phase)

    @property
    def phase(self):
        """Get/Set Phase of the wavefront. Taking/returning
        an array of shape=(npix, npix).

        :getter: Phase of the wavefront.
        :setter: Loads given array as phase of the wavefront,
            leaving amplitude unchanged.

        :type: np.array

        """
        return np.angle(self.wavefront)

    @phase.setter
    def phase(self, phase):
        self.wavefront = self.amplitude * np.exp(1.j * phase)

    def to_fits(self, path):
        """A method to serialize wavefront to fits. Amplitude and phase are saved
        separately to seperate tables

        :param path: path to the file to which wavefront is saved
        :type path: str

        **Example**

        >>> import astropy.units as u
        >>> import pyoptica as po
        >>>
        >>> wavelength = 500 * u.nm
        >>> pixel_scale = 22 * u.um
        >>> npix = 1024
        >>> wf = po.Wavefront(wavelength, pixel_scale, npix)
        >>> file_path = 'path/to/your/file.fits'
        >>> wf.to_fits(file_path)

        """
        amplitude = self.amplitude
        phase = self.phase
        hdu_list = fits.HDUList()
        hdr = fits.Header()
        hdr['WVL'] = self.wavelength.to(u.nm).value,
        hdr['PIXSCALE'] = self.pixel_scale.to(u.um).value,
        hdr['NPIX'] = self.npix
        hdu_list.append(fits.PrimaryHDU(header=hdr))
        hdu_list.append(fits.ImageHDU(amplitude, header=None, name='AMP',
                                      do_not_scale_image_data=True))
        hdu_list.append(fits.ImageHDU(phase, header=None, name='PHS',
                                      do_not_scale_image_data=True))
        hdu_list.writeto(path, overwrite=True)
        self.logger.debug(f"Saved wavefront to {path}")

    @classmethod
    def from_fits(cls, path):
        """A method to load wavefront from a fits file.

        :param path: path to the file to be loaded.
        :type path: str
        :return: loaded wavefront
        :rtype: pyoptica.Wavefront

        **Example**

        >>> import pyoptica as po
        >>>
        >>> file_path = 'path/to/your/file.fits'
        >>> wf = po.Wavefront.from_fits(file_path)

        """
        hdu_list = fits.open(path, memmap=True)
        wavelength = hdu_list[0].header['WVL'] * u.nm
        pixel_scale = hdu_list[0].header['PIXSCALE'] * u.um
        npix = hdu_list[0].header['NPIX']
        wf = Wavefront(wavelength, pixel_scale, npix)
        wf.amplitude = hdu_list['AMP'].data
        wf.phase = hdu_list['PHS'].data
        cls.logger.debug(f"Loaded wavefront from {path}")
        return wf

    def _apply_phase_plotting_threshold(self):
        # Phase is plotted only for non-zero intensity values
        avg_i = np.average(self.intensity[self.intensity > 0])
        no_phase_indices = np.where(
            self.intensity < avg_i * 0.01
        )
        phase = self.phase
        phase[no_phase_indices] = np.nan
        return phase
