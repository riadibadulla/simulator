import math

import astropy.units as u
import numpy as np
import pyoptica as po
from matplotlib import pyplot as plt
import seaborn as sns
from torch import tensor
from skimage import io
from skimage.color import rgb2gray
from scipy import signal
import torch

class Filter(po.BaseOpticalElement):

    def __init__(self, filter=None):
        if filter:
            self.filter = torch.fft.fftshift(torch.fft.fft2(filter))

    def set_filter(self,filter):
        self.filter = torch.fft.fftshift(torch.fft.fft2(filter))

    def amplitude_transmittance(self, wavefront):
        return self.filter

    def phase_transmittance(self, wavefront):
        #TODO: may need to edit phase getter
        return torch.ones_like(torch.tensor(wavefront.phase))

class Optics_simulation:

    def __init__(self,number_of_pixels=300):

        self.wavelength = 500 * u.nm
        self.npix = number_of_pixels
        self.f = 6 * u.cm
        self.pixel_scale = 10 *u.um
        self.wf = po.Wavefront(self.wavelength, self.pixel_scale, self.npix)
        self.r = 1.5 * u.mm
        self.lens = po.ThinLens(2*self.r, self.f)
        self.fs = po.FreeSpace(self.f)
        self.filter = Filter()

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
        # wf_imaged = self.wf * self.fs * self.lens * self.fs * self.filter * self.fs * self.lens * self.fs
        wf_imaged = self.wf * self.fs * self.lens * self.fs * self.filter * self.fs * self.lens * self.fs
        return wf_imaged.amplitude

    def __pad(self,large,small,padding_size):
        small = torch.nn.functional.pad(small, (padding_size,padding_size,padding_size,padding_size))
        if small.shape != large.shape:
            small = torch.nn.functional.pad(small, (0,1,0,1))
        return large,small

    def process_inputs(self,img, kernel):
        if img.shape==kernel.shape:
            return img, kernel

        size_of_image = img.shape[0]
        size_of_kernel = kernel.shape[0]
        padding_size = abs(size_of_image - size_of_kernel) // 2
        if size_of_image > size_of_kernel:
            img, kernel = self.__pad(img,kernel,padding_size)
        else:
            kernel, img = self.__pad(kernel,img,padding_size)
        return img, kernel


    def optConv2d(self, img,kernel,pseudo_negativity=False):
        img, kernel = self.process_inputs(img, kernel)
        if pseudo_negativity:
            #TODO: make it tensor
            pos, neg = np.maximum(kernel, 0), np.maximum(kernel * (-1), 0)
            output_pos = self.convolution_4F(img, pos)
            output_neg = self.convolution_4F(img, neg)
            result = output_pos - output_neg
        else:
            #TODO: adapt size of inputs to npix
            result = self.convolution_4F(img, kernel)
        result = torch.fft.fftshift(result)
        output_final = self.no_convolution_4F(result)
        return output_final

# if __name__ == '__main__':
#     img = io.imread("noisy.jpg")
#     img = rgb2gray(img)
#     img = torch.tensor(img)
#     optics = Optics_simulation(img.shape[0])
#     kernel = np.array(
#         [[1, 4, 7, 4, 1], [4, 16, 26, 16, 4], [7, 26, 41, 26, 7], [4, 16, 26, 16, 4], [1, 4, 7, 4, 1]])
#     kernel = torch.tensor(kernel)
#     output = optics.optConv2d(img, kernel, False)
#     plt.imshow(output, cmap='gray')
#     plt.show()
#     plt.imshow(signal.fftconvolve(img, kernel, mode="same"), cmap='gray')
#     plt.show()
