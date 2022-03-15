import math

import astropy.units as u
import numpy as np
import pyoptica as po
from matplotlib import pyplot as plt
import seaborn as sns
from torch import tensor
from skimage import io
from skimage.transform import rescale, resize, downscale_local_mean
from skimage.color import rgb2gray
from scipy import signal
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.modules.activation import ReLU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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
        return torch.ones_like(wavefront.phase)

class Optics_simulation:

    def __init__(self,number_of_pixels=28):

        self.wavelength = 532 * u.nm
        self.npix = number_of_pixels
        self.f = 10 * u.mm
        self.pixel_scale = math.sqrt(532*10**(-11)/number_of_pixels)* u.m
        self.wf = po.Wavefront(self.wavelength, self.pixel_scale, self.npix)
        self.r = 2.5 * u.mm
        self.lens = po.ThinLens(self.r, self.f)
        self.fs = po.FreeSpace(self.f,wavefront=self.wf)
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


    def optConv2d(self, img,kernel,pseudo_negativity=True):
        img, kernel = self.process_inputs(img, kernel)
        if pseudo_negativity:
            relu = ReLU()
            pos, neg = relu(kernel), relu(kernel * (-1))

            output_pos = self.convolution_4F(img, pos)
            output_neg = self.convolution_4F(img, neg)
            # output_pos = torch.fft.ifft(torch.fft.fft2(img)*torch.fft.fft2(pos))
            # output_neg = torch.fft.ifft(torch.fft.fft2(img)*torch.fft.fft2(neg))
            result = torch.sub(output_pos,output_neg)
        else:
            result = self.convolution_4F(img, kernel)
        result = torch.fft.fftshift(result)
        result = self.no_convolution_4F(result)
        return result


if __name__ == '__main__':
    img = io.imread("mnist-test.jpg", as_gray=True)
    img = resize(img, (28,28),anti_aliasing=True)/255
    plt.imshow(img, cmap='gray')
    plt.show()
    img1 = Variable(torch.tensor(img), requires_grad=True).to(device)
    optics = Optics_simulation(img1.shape[0])
    kernel = np.array(
        [[1, 4, 7, 4, 1], [4, 16, 26, 16, 4], [7, 26, 41, 26, 7], [4, 16, 26, 16, 4], [1, 4, 7, 4, 1]])/26
    kernel = Variable(torch.tensor(kernel, dtype=torch.float64), requires_grad=True).to(device)
    output = optics.optConv2d(img1, kernel, True)
    plt.imshow(output.cpu().detach().numpy(), cmap='gray')
    plt.show()
    plt.imshow(signal.correlate(img, kernel.cpu().detach().numpy(), mode="same"), cmap='gray')
    plt.show()
