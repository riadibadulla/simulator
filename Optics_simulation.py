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

    def process_inputs(self,img, sample_kernel):

        size_of_image = img.shape[0]
        size_of_kernel = sample_kernel.shape[0]
        padding_size = abs(size_of_image - size_of_kernel) // 2

        if size_of_image>size_of_kernel:
            padded_kernel = np.pad(np.array(sample_kernel), padding_size)
            if padded_kernel.shape != img.shape:
                padded_kernel = np.pad(padded_kernel, ((0,1),(0,1)))
        else:
            #TODO: Needs to be refactored
            padded_image = np.pad(np.array(img), padding_size)
            if padded_image.shape != sample_kernel.shape:
                padded_image = np.pad(padded_image, ((0,1),(0,1)))
        return img, padded_kernel


    def optConv2d(self, img,kernel,pseudo_negativity=True):
        # if pseudo_negativity:
        #     pos, neg = np.maximum(kernel, 0), np.maximum(kernel * (-1), 0)
        #     output_pos = fft_based_convolution(INPUT_IMAGE, pos)
        #     output_pos = optics.no_convolution_4F(output_pos)
        #     output_neg = fft_based_convolution(INPUT_IMAGE, neg)
        #     optics = Optics_simulation(img.shape[0])
        #     output_neg = optics.no_convolution_4F(output_neg)
        #     output_fin = output_pos - output_neg
        # else:
        img, kernel = self.process_inputs(img, kernel)
        #TODO: adapt size of inputs to npix
        result = self.convolution_4F(img, kernel)
        result = np.fft.fftshift(result)
        output_final = self.no_convolution_4F(result)
        return output_final