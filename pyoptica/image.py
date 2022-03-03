import astropy.units as u
import numpy as np
from astropy.io import fits

from . import distribution


class Image(distribution.Distribution):
    """Class representing an Image -- result of the imaging operation.
    In addition to the image it stores information regarding wavelength,
    pixel sampling, and dimensions of the image distribution.

    :param wavelength: Wavelength of the simulated image
    :type wavelength: astropy.Quantity of type length
    :param pixel_scale: Sampling of the image
    :type pixel_scale: astropy.Quantity of type length
    :param npix: Number of pixels in the image array
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
    >>> image = po.Image(wavelength, pixel_scale, npix)

    """
    PLOTTING_OPTIONS = {
        'image': dict(x='x', y='y', title='Image')
    }
    @u.quantity_input(wavelength=u.m, pixel_scale=u.m)
    def __init__(self, wavelength, pixel_scale, npix):
        super().__init__(wavelength, pixel_scale, npix)
        self.image = np.ones((npix, npix), dtype=np.float)
        self.logger.debug(f"Created {self}")

    def to_fits(self, path):
        """A method to serialize image to fits.

        :param path: path to the file to which image is saved
        :type path: str

        **Example**

        >>> import astropy.units as u
        >>> import pyoptica as po
        >>>
        >>> wavelength = 500 * u.nm
        >>> pixel_scale = 22 * u.um
        >>> npix = 1024
        >>> image = po.Image(wavelength, pixel_scale, npix)
        >>> file_path = 'path/to/your/file.fits'
        >>> image.to_fits(file_path)

        """
        image = self.image
        hdu_list = fits.HDUList()
        hdr = fits.Header()
        hdr['WVL'] = self.wavelength.to(u.nm).value,
        hdr['PIXSCALE'] = self.pixel_scale.to(u.um).value,
        hdr['NPIX'] = self.npix
        hdu_list.append(fits.PrimaryHDU(header=hdr))
        hdu_list.append(fits.ImageHDU(image, header=None, name='IMG',
                                      do_not_scale_image_data=True))
        hdu_list.writeto(path, overwrite=True)
        self.logger.debug(f"Saved wavefront to {path}")

    @classmethod
    def from_fits(cls, path):
        """A method to load image from a fits file.

        :param path: path to the file to be loaded.
        :type path: str
        :return: loaded image
        :rtype: pyoptica.Image

        **Example**

        >>> import pyoptica as po
        >>>
        >>> file_path = 'path/to/your/file.fits'
        >>> image = po.Image.from_fits(file_path)

        """
        hdu_list = fits.open(path, memmap=True)
        wavelength = hdu_list[0].header['WVL'] * u.nm
        pixel_scale = hdu_list[0].header['PIXSCALE'] * u.um
        npix = hdu_list[0].header['NPIX']
        image = Image(wavelength, pixel_scale, npix)
        image.image = hdu_list['IMG'].data
        cls.logger.debug(f"Loaded image from {path}")
        return image

