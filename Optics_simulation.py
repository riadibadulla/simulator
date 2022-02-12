import astropy.units as u
import numpy as np
import pyoptica as po
from matplotlib import pyplot as plt

class Optics_simulation:

    def __init__(self):
        self.wavelength = 500 * u.nm
        self.pixel_scale = 10 * u.um
        self.npix = 450
        # 300
        self.na = 0.35
        self.coherence_factor = 0

        self.f = 9 * u.cm
        self.axis_unit = u.mm
        self.wf = po.Wavefront(self.wavelength, self.pixel_scale, self.npix)
        self.r = 2 * u.mm
        self.lens = po.ThinLens(2*self.r, self.f)
        self.fs = po.FreeSpace(self.f)

    def rgb2gray(self, rgb):
        img = np.dot(rgb[..., :3], [0.299, 0.587, 0.114])
        return img

    def plot_wavefront(self, wavefront, amplitude):
        fig = wavefront.plot(amplitude=amplitude, fig_options=dict(figsize=(5, 5), dpi=130))
        fig[0].show()

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
        convoluted_matrix = np.multiply(wf_in_frequency_domain.wavefront, kernel)
        self.wf.wavefront = convoluted_matrix
        wf_imaged = self.make_fourier_engine(self.wf)
        self.plot_wavefront(wf_imaged, "default")
        return wf_imaged.amplitude


