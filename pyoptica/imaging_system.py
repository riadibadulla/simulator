import astropy.units as u
import numpy as np

from . import logging
from . import plotting
from . import utils
from . import zernikes
from .image import Image
from .wavefront import Wavefront


class ImagingSystem(logging.LoggerMixin, plotting.Plottable):
    """ A class representing an imaging systems. It is based on reasoning
    presented in [1] and [2] (numerical implementation suggested in
    Chapter 7 [3]). An optical system composed of many optical elements can be
    described by one function that _aggregates_ all properties of them all.
    In case of coherent imaging the, the image is found based on
    (eqs. 5.37-5.40 in [1], the naming convention follows that given in
    [1] *Introduction to Fourier Optics* by J.W. Goodman)

    It is assumed that the system is isoplanatic (the aberrations are constant
    through out the field)

    *THE IMAGING SYSTEM HAS A CIRCULAR APERTURE!*

    :param wavelength: Wavelength used in simulations
    :type wavelength: astropy.Quantity of type length
    :param pixel_scale: Sampling of the wavefront
    :type pixel_scale: astropy.Quantity of type 1/length
    :param npix: Number of pixels in the wavefront array
    :type npix: int
    :param coherence_factor: complex coherence factor
    :type coherence_factor: float
    :param m: transverse magnification of the optical system
    :type m: float
    :param na: numerical aperture defined as :math:`NA = sin\theta_{max}`
    :type na: float
    :param n_o: refractive index on the object side
    :type n_o: float
    :param n_i: refractive index on the image side
    :type n_i: float

    **Example**

    >>> import astropy.units as u
    >>> import pyoptica as po
    >>>
    >>> wavelength = 1 * u.um
    >>> pixel_scale = 10 * u.mm
    >>> npix = 1024
    >>> coh_factor = 1  # Fully coherent
    >>> img_sys = po.ImagingSystem(wavelength, pixel_scale, npix, coh_factor)

    **References**

    [1] Joseph W. Goodman (2004) -
    *Introduction to Fourier Optics*, W. H. Freeman

    [2] Kedar Khare (2016) -
    *Fourier Optics and Computational Imaging*, Wiley&Sons Ltd.

    [3] David Voelz (2011) -
    *Computational Fourier Optics Matlab Tutorial*, Spie Press

    [4] M. Born, E. Wolf (1980) -
    *Principles of Optics*, Pergamon Press (Oxford UK)

    """
    PLOTTING_OPTIONS = {
        'psf': dict(
            x='x', y='y', title='|PSF|', z_function=lambda x: np.abs(x)
        ),
        'atf': dict(
            x='f', y='g', title='|ATF|', z_function=lambda x: np.abs(x)
        ),
        'ptf': dict(x='f', y='g', title='PTF'),
        'mtf': dict(x='f', y='g', title='MTF'),
        'otf': dict(
            x='f', y='g', title='|OTF|', z_function=lambda x: np.abs(x)
        ),
        'wavefront': dict(
            colorbar_title='[rad]', vmin=-np.pi, vmax=np.pi,
            bar_ticks=[-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
            bar_ticks_labels=[
                r'$-\pi$',
                r'$-\frac{\pi}{2}$',
                r'$0$',
                r'$\frac{\pi}{2}$',
                r'$\pi$'
            ],
            x='f', y='g', cmap='hsv', title='Wavefront (aberrations)',
            z_function='_aberrations_plot_special'
        )
    }

    @u.quantity_input(wavelength=u.m, pixel_scale=u.m)
    def __init__(self, wavelength, pixel_scale, npix, coherence_factor, m=1,
                 na=1, n_o=1, n_i=1):
        self.wavelength = wavelength
        self.pixel_scale = pixel_scale
        self.npix = npix
        self.coherence_factor = coherence_factor
        self.m = m
        self.na = na
        self.n_o = n_o
        self.n_i = n_i

        self.x, self.y = utils.mesh_grid(npix, pixel_scale)

        self._atf = np.ones((npix, npix))
        self._otf = np.ones((npix, npix))

        self._zernike_coefs = dict()
        self._zernike_convention = None
        self._apodization = None
        self._system_modified = True

    def __setattr__(self, key, value):
        # I really like this!
        if key != '_system_modified':
            self._system_modified = True
        super().__setattr__(key, value)

    @property
    def f(self):
        f = utils.space2freq(self.x)
        return f

    @property
    def g(self):
        g = utils.space2freq(self.y)
        return g

    def load_zernikes(self, zernike_coefs, convention):
        """
        Loads zernike coefficients represented as a dict: `{index: val}`.
        The coefficients are expected to be in RADIANS!

        :param zernike_coefs: zernikes to be loaded
        :type zernike_coefs: dict
        :param convention: the way zernikes are represented (`mn`, `osa`, or `noll`)
        :type convention: str
        :raises: ValueError

        **Example**

        >>> import astropy.units as u
        >>> import pyoptica as po
        >>>
        >>> wl = 1 * u.um
        >>> pixel_scale = 10 * u.mm
        >>> npix = 1024
        >>> coh_factor = 1  # Fully coherent
        >>> img_sys = po.ImagingSystem(wl, pixel_scale, npix, coh_factor)
        >>>
        >>> zernike_coefs = {1: 0.2, 2: 0.3, 7: 0.1, 10: 0.01}  # OSA
        >>> img_sys.load_zernikes(zernike_coefs)
        """
        self._zernike_coefs = zernike_coefs
        self._zernike_convention = convention


    def _calc_atf(self):
        """Calculates Amplitude Transfer Function H based on the current state
        of the object (numerical aperture, magnification, aberrations, and
        apodization).

        """
        H_ideal = self._calc_diffraction_limited_atf()
        H = self._apply_radiometric_correction(H_ideal)
        H_with_aberrations = self._apply_aberrations(H)
        self._atf = H_with_aberrations

    def _calc_otf(self):
        """ Calculates Optical Transfer Function based on the current state
        of the object. It is calculated as a normalized autocorrelation of the
        Amplitude Transfer Function.
        """
        H_fft = utils.fft(self._atf)
        H_fft_abs_squared = np.abs(H_fft) ** 2
        otf = utils.ifft(H_fft_abs_squared)
        i, j = otf.shape
        otf /= otf[i // 2, j // 2]
        rho = np.hypot(self.f, self.g)
        otf[rho > 2 * self.na / self.wavelength] = 0
        self._otf = otf

    def calculate(self):
        """Precalculates both Amplitude and Optical Transfer Functions
        (including aberrations, magnification, apodization).

        This method must be called always after manipulating the imaging
        system otherwise the results will be inaccurate.

        **Example**

        >>> import astropy.units as u
        >>> import pyoptica as po
        >>>
        >>> wavelength = 1 * u.um
        >>> pixel_scale = 10 * u.mm
        >>> npix = 1024
        >>> mu = 1
        >>> img_sys = po.ImagingSystem(wavelength, pixel_scale, npix, mu)
        >>> img_sys.calculate()
        """
        self._calc_atf()
        self._calc_otf()
        self._system_modified = False

    @property
    def atf(self):
        """Amplitude Transfer Function (Goodman naming convention).

        **Example**

        >>> import astropy.units as u
        >>> import pyoptica as po
        >>>
        >>> wl = 1 * u.um
        >>> pixel_scale = 10 * u.mm
        >>> npix = 1024
        >>> coh_factor = 1
        >>> img_sys = po.ImagingSystem(wl, pixel_scale, npix, coh_factor)
        >>> img_sys.calculate()  # System must be precalculated first!
        >>> img_sys.atf

        """
        self._warn_if_modified()
        return self._atf

    @property
    def wavefront(self):
        """Wavefront (aberrations) of the imaging system.

        **Example**

        >>> import astropy.units as u
        >>> import pyoptica as po
        >>>
        >>> wl = 1 * u.um
        >>> pixel_scale = 10 * u.mm
        >>> npix = 1024
        >>> coh_factor = 1
        >>> img_sys = po.ImagingSystem(wl, pixel_scale, npix, coh_factor)
        >>> img_sys.calculate()  # System must be precalculated first!
        >>> img_sys.wavefront

        """
        self._warn_if_modified()
        return np.angle(self._atf)

    @property
    def otf(self):
        """Optical Transfer Function (Goodman naming convention).

        **Example**

        >>> import astropy.units as u
        >>> import pyoptica as po
        >>>
        >>> wl = 1 * u.um
        >>> pixel_scale = 10 * u.mm
        >>> npix = 1024
        >>> coh_factor = 1
        >>> img_sys = po.ImagingSystem(wl, pixel_scale, npix, coh_factor)
        >>> img_sys.calculate()  # System must be precalculated first!
        >>> img_sys.otf

        """
        self._warn_if_modified()
        return self._otf

    @property
    def psf(self):
        """Amplitude Point-Spread Function (Goodman naming
        convention).

        **Example**

        >>> import astropy.units as u
        >>> import pyoptica as po
        >>>
        >>> wl = 1 * u.um
        >>> pixel_scale = 10 * u.mm
        >>> npix = 1024
        >>> coh_factor = 1
        >>> img_sys = po.ImagingSystem(wl, pixel_scale, npix, coh_factor)
        >>> img_sys.calculate()  # System must be precalculated first!
        >>> img_sys.psf

        """
        return utils.ifft(self.atf)

    @property
    def mtf(self):
        """Modulation Transfer Function (MTF = |OTF|)
         (Goodman naming convention).

        **Example**

        >>> import astropy.units as u
        >>> import pyoptica as po
        >>>
        >>> wl = 1 * u.um
        >>> pixel_scale = 10 * u.mm
        >>> npix = 1024
        >>> coh_factor = 1
        >>> img_sys = po.ImagingSystem(wl, pixel_scale, npix, coh_factor)
        >>> img_sys.calculate()  # System must be precalculated first!
        >>> img_sys.mtf
        """
        return np.abs(self.otf)

    @property
    def ptf(self):
        """Phase Transfer Function (PTF = arg(OTF))
         (Goodman naming convention).

        **Example**

        >>> import astropy.units as u
        >>> import pyoptica as po
        >>>
        >>> wl = 1 * u.um
        >>> pixel_scale = 10 * u.mm
        >>> npix = 1024
        >>> coh_factor = 1
        >>> img_sys = po.ImagingSystem(wl, pixel_scale, npix, coh_factor)
        >>> img_sys.calculate()  # System must be precalculated first!
        >>> img_sys.ptf
        """
        return np.angle(self.otf)

    def _warn_if_modified(self):
        if self._system_modified:
            self.logger.warning(
                'The system has been modified, however, it has not been '
                'recalculated. The results may be inaccurate. '
                'Please rerun `calculate` to obtain results that represent '
                'the current state of the system.')

    def _calc_diffraction_limited_atf(self):
        """
        The diffraction limited Amplitude Transfer Function is calculated:

        :math:`\left\{\begin{matrix} 1, & \text{if } f^2 +g^2 < f_0^2 \\ 0, & \text{otherwise} \end{matrix}\right.`

        :return: Diffraction limited Amplitude Transfer Function
        :rtype: np.array

        """
        rho_squared = self.f ** 2 + self.g ** 2
        H = np.where(rho_squared < (self.na / self.wavelength) ** 2, 1, 0)
        return H

    def _apply_radiometric_correction(self, atf_ideal):
        """ If there is magnification (or reduction) in the system, the
        distribution of light entering the system will be different than the
        distribution leaving it. In order to include that in the Amplitude
        Transfer Function H, a radiometric correction has to be applied
        (following M. Born and E. Wolf [4]):

        :math:`\left ( \frac{1 - \lambda^2(f^2 +g^2)M^2}{1 - \lambda^2(f^2 +g^2)}\right )^{0.25}`

        :param atf_ideal: diffraction limited Amplitude Transfer Function
        :type atf_ideal: np.array
        :return: Amplitude Transfer Function with radiometric correction
        :rtype: np.array

        """
        rho_squared = self.f ** 2 + self.g ** 2
        numerator = 1 - (
                self.wavelength ** 2 * rho_squared) * self.m ** 2 / self.n_o ** 2
        denominator = 1 - (self.wavelength ** 2 * rho_squared) / self.n_i ** 2
        r = numerator / denominator
        H = np.power(atf_ideal * r, 0.25)
        H = H.to(u.dimensionless_unscaled).value
        #  Maybe it would make more sense to raise an exception if NA > n
        H[~np.isfinite(H)] = 0
        return H

    def _apply_aberrations(self, H):
        """Apply aberrations from `self._zernike_coefs` to the amplitude transfer
        function (H):

        :math:`H=H_d \cdot \exp(2 \pi i W)`

        Where `H_d` is the diffraction limited transfer function, and `W` the
        calculated wavefront.

        :param H: The diffraction limited H
        :type H: np.array
        :return: amplitude transfer function with aberrations
        :rtype: np.array

        """
        rho, theta = utils.cart2pol(self.f, self.g)
        rho = rho.to(1 / u.m) / (self.na / self.wavelength).to(1 / u.m)
        wf = zernikes.construct_wavefront(
            self._zernike_coefs, self._zernike_convention, rho, theta
        )
        return H * np.exp(2.j * np.pi * wf)

    def image_wavefront(self, wf):
        """ A method to image given wavefront. There are three distinct cases:

        +--------------------+------------------------------+---------------------------------+
        | Coherence          | Fundamental Quantity         | Transfer Function               |
        +====================+==============================+=================================+
        | Fully coherent     | Input wavefront field        | Amplitude Transfer Function     |
        +--------------------+------------------------------+---------------------------------+
        | Fully incoherent   | Intensity of input wavefront | Optical Transfer Function       |
        +--------------------+------------------------------+---------------------------------+
        | Partially coherent | Mutual intensity             | Transmission Cross-Coefficients |
        +--------------------+------------------------------+---------------------------------+

        :param wf: Input wavefront
        :type wf: pyoptica.Wavefront
        :return: image in the image plane
        :rtype: pyoptica.Image

        **Example**

        >>> import astropy.units as u
        >>> import pyoptica as po
        >>>
        >>> wl = 1 * u.um
        >>> pixel_scale = 10 * u.mm
        >>> npix = 1024
        >>> coh_factor = 1  # Fully coherent
        >>> img_sys = ImagingSystem(wl, pixel_scale, npix, coh_factor)
        >>> img_sys.calculate()  # Remember to call `calculate`!
        >>> wf = Wavefront(wl, 1/pixel_scale/npix, npix)
        >>> imaged_wf = img_sys.image_wavefront(wf)
        >>> imaged_wf

        """
        self._check_wavefront_compatibility(wf)

        if self.coherence_factor == 1:
            return self._image_wavefront_fully_coherent(wf)
        elif self.coherence_factor == 0:
            return self._image_wavefront_fully_incoherent(wf)
        elif 0 < self.coherence_factor < 1:
            return self._image_wavefront_partially_coherent(wf)
        else:
            raise ValueError('Complex coherence factor is limited to [0, 1]!')

    def _image_wavefront_fully_coherent(self, wf):
        """ Images given wavefront assuming fully coherent light. The image
        is obtained as squared absolute value of convolution of input wavefront
         field and ATF of the system. For computational efficiency fourier
        transform convolution theorem is employed.

        :param wf: wavefront to be imaged
        :type wf: pyoptica.Wavefront
        :return: image
        :rtype: pyoptica.Image
        """
        G_g = utils.fft(wf.wavefront)
        G_i = G_g * self.atf
        u_i = utils.ifft(G_i)
        pix_scale = self.pixel_scale * self.m
        image = Image(self.wavelength, pix_scale, self.npix)
        image.image = np.abs(u_i) ** 2
        return image

    def _image_wavefront_fully_incoherent(self, wf):
        """ Images given wavefront assuming fully incoherent light. The image
        is obtained as a convolution of normalized intensity of the input
        wavefront and OTF of the system. For computational efficiency fourier
        transform convolution theorem is employed.

        :param wf: wavefront to be imaged
        :type wf: pyoptica.Wavefront
        :return: image
        :rtype: pyoptica.Image
        """
        I_g = wf.intensity
        I_g /= I_g.max()
        G_g = utils.fft(wf.intensity)
        G_i = G_g * self.otf
        I_i = utils.ifft(G_i)
        pix_scale = self.pixel_scale * self.m
        image = Image(self.wavelength, pix_scale, self.npix)
        image.image = np.real(I_i)
        return image

    def _image_wavefront_partially_coherent(self, wf):
        raise NotImplementedError(
            'Partial coherence simulation has not been implemented yet.')

    def _check_wavefront_compatibility(self, wf):
        """Checks if wavefront is compatible with the system
        :param wf: Wavefront
        :type wf: pyoptica.Wavefront
        :raises: RuntimeError

        """
        if self.npix != wf.npix:
            raise RuntimeError('Imaging system npix != wavefront npix'
                               f'{self.npix} != {wf.npix}')
        if self.wavelength != wf.wavelength:
            raise RuntimeError(
                'Imaging system wavelength != wavefront wavelength'
                f'{self.wavelength} != {wf.wavelength}')
        if self.pixel_scale != wf.pixel_scale:
            raise RuntimeError(
                'Imaging system wavelength != wavefront wavelength'
                f'{self.pixel_scale} != {wf.pixel_scale}')

    def _aberrations_plot_special(self):
        rho, theta = utils.cart2pol(self.f, self.g)
        rho = rho.to(1 / u.m) / (self.na / self.wavelength).to(1 / u.m)
        wavefront = self.wavefront
        wavefront[rho > 1] = np.nan
        return wavefront
