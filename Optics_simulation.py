import math

import astropy.units as u
import numpy as np
import pyoptica as po
from matplotlib import pyplot as plt
import seaborn as sns

class Optics_simulation:

    def __init__(self,number_of_pixels):

        self.wavelength = 500 * u.nm
        self.npix = number_of_pixels
        self.na = 0.35
        self.coherence_factor = 0
        self.f = 3 * u.cm
        self.axis_unit = u.mm
        self.pixel_scale = 122.5/math.sqrt(number_of_pixels)*u.um

        self.img_sys = po.ImagingSystem(self.wavelength, self.pixel_scale, self.npix, self.coherence_factor)
        self.img_sys.calculate()

        self.wf = po.Wavefront(self.wavelength, self.pixel_scale, self.npix)
        self.r = 4 * u.mm
        self.lens = po.ThinLens(2*self.r, self.f)
        self.fs = po.FreeSpace(self.f)

    def rgb2gray(self, rgb):
        img = np.dot(rgb[..., :3], [0.299, 0.587, 0.114])
        return img

    def plot_wavefront(self, wavefront, amplitude):
        fig = wavefront.plot(amplitude=amplitude, fig_options=dict(figsize=(5, 5), dpi=130))
        fig[0].show()
        ax = sns.heatmap(wavefront.amplitude, annot=True)
        plt.show()

    def make_4F_engine(self, waveform):
        return waveform * self.fs * self.lens * self.fs * self.fs * self.lens * self.fs

    def make_fourier_engine(self, waveform):
        return waveform * self.fs * self.lens * self.fs

    def no_convolution_4F(self, img):
        self.wf.amplitude = img
        wf_imaged = self.make_4F_engine(self.wf)
        return wf_imaged.amplitude

    def convolution_4F(self, img, kernel):
        self.wf.amplitude = img
        wf_in_frequency_domain = self.make_fourier_engine(self.wf)
        self.plot_wavefront(wf_in_frequency_domain, "default")
        print(wf_in_frequency_domain.amplitude)
        convoluted_matrix = np.multiply(wf_in_frequency_domain.wavefront, kernel)
        self.wf.wavefront = convoluted_matrix
        wf_imaged = self.make_fourier_engine(self.wf)
        # self.plot_wavefront(wf_imaged, "default")
        return wf_imaged.amplitude



if __name__=='__main__':
    np.random.seed(2022)
    img = np.random.random(size=(10, 10))

    #NORMAL FFT
    img_fr = np.fft.fftshift(np.fft.fft2(img))
    sns.heatmap(abs(img_fr), annot=True)
    plt.show()

    #OPTICS
    wavelength = 500 * u.nm
    npix = 10
    f = 12.8 * u.m
    pixel_scale = 0.8 * u.mm
    wf = po.Wavefront(wavelength, pixel_scale, npix)
    r = 4 * u.mm
    lens = po.ThinLens(r, f)
    fs = po.FreeSpace(f)

    wf.amplitude = img
    wf_in_frequency_domain = wf * fs * lens * fs
    ax = sns.heatmap(wf_in_frequency_domain.amplitude, annot=True)
    plt.show()