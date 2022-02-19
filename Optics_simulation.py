import math

import astropy.units as u
import numpy as np
import pyoptica as po
from matplotlib import pyplot as plt
import seaborn as sns

class Filter(po.BaseOpticalElement):

    def __init__(self, filter=None):
        if filter:
            self.filter = np.fft.fftshift(np.fft.fft2(filter))

    def set_filter(self,filter):
        self.filter = np.fft.fftshift(np.fft.fft2(filter))

    def amplitude_transmittance(self, wavefront):
        return self.filter

    def phase_transmittance(self, wavefront):
        return np.ones_like(wavefront.phase)

class Optics_simulation:

    def __init__(self,number_of_pixels):

        self.wavelength = 500 * u.nm
        self.npix = number_of_pixels
        self.f = 6 * u.cm
        self.pixel_scale = 10 *u.um
        self.wf = po.Wavefront(self.wavelength, self.pixel_scale, self.npix)
        self.r = 1.5 * u.mm
        self.lens = po.ThinLens(2*self.r, self.f)
        self.fs = po.FreeSpace(self.f)
        self.filter = Filter()

    def rgb2gray(self, rgb):
        img = np.dot(rgb[..., :3], [0.299, 0.587, 0.114])
        return img

    def plot_wavefront(self, wavefront, amplitude):
        # fig = wavefront.plot(amplitude=amplitude, fig_options=dict(figsize=(5, 5), dpi=130))
        # fig[0].show()
        plt.imshow(wavefront.amplitude)
        plt.show()

    def make_4F_engine(self, waveform):
        waveform * self.fs * self.lens * self.fs * self.fs * self.lens * self.fs

    def make_fourier_engine(self, waveform):
        return waveform * self.fs * self.lens * self.fs

    def no_convolution_4F(self, img):
        self.wf.amplitude = img
        wf_imaged = self.wf * self.fs * self.lens * self.fs * self.fs * self.lens * self.fs
        return wf_imaged.amplitude

    def convolution_4F(self, img, kernel):
        self.wf.amplitude = img
        self.filter.set_filter(kernel)
        wf_imaged = self.wf * self.fs * self.lens * self.fs * self.filter * self.fs * self.lens * self.fs
        return wf_imaged.amplitude

