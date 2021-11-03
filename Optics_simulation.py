import astropy.units as u
import numpy as np
import pyoptica as po
from matplotlib import pyplot as plt

class Optics_simulation:

    def __init__(self):
        self.wavelength = 500 * u.nm
        self.pixel_scale = 0.1 * u.um
        self.npix = 300
        self.na = 0.35
        self.coherence_factor = 0
        self.f= 0.0006 * u.cm
        self.axis_unit = u.mm
        self.wf = po.Wavefront(self.wavelength, self.pixel_scale, self.npix)
        self.r = 2 * u.mm
        self.lens = po.ThinLens(2*self.r, self.f)
        self.fs = po.FreeSpace(self.f)

    def rgb2gray(self, rgb):
        img = np.dot(rgb[..., :3], [0.299, 0.587, 0.114])
        return img

    def modulate_signal(self, image_dir):
        img = plt.imread(image_dir)
        target = self.rgb2gray(img)
        self.wf.amplitude = target

    def plot_wavefront(self, wavefront, amplitude):
        fig = wavefront.plot(amplitude=amplitude, fig_options=dict(figsize=(5, 5), dpi=130))
        fig[0].show()

    def make_4F_engine(self, waveform):
        return waveform * self.fs * self.lens * self.fs * self.fs * self.lens * self.fs

    def make_fourier_engine(self, waveform):
        return waveform * self.fs * self.lens * self.fs

    def no_convolution_4F(self, input_image_link):
        self.modulate_signal(input_image_link)
        self.plot_wavefront(self.wf, dict(vmax=1))
        # print(fs.calc_optimal_distance(wf))
        wf_imaged = self.make_4F_engine(self.wf)
        print(type(wf_imaged))
        self.plot_wavefront(wf_imaged, dict(vmax=1))

    def convolution_4F(self, input_image_link,initial_kernel):
        self.modulate_signal(input_image_link)
        self.plot_wavefront(self.wf, dict(vmax=1))
        wf_in_frequency_domain = self.make_fourier_engine(self.wf)
        self.plot_wavefront(wf_in_frequency_domain, dict(vmax=1))

        amplitude_in_fourier = wf_in_frequency_domain.amplitude
        convoluted_matrix = np.multiply(amplitude_in_fourier, initial_kernel)
        self.wf.amplitude = convoluted_matrix
        wf_imaged = self.make_fourier_engine(self.wf)
        self.plot_wavefront(wf_imaged, dict(vmax=0.1))


