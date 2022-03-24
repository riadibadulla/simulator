import math

import numpy as np
# from matplotlib import pyplot as plt
# from skimage import io
# from skimage.transform import rescale, resize, downscale_local_mean
# from scipy import signal
import torch
# from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.modules.activation import ReLU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
import utils

class Optics_simulation:

    def __init__(self,number_of_pixels=28):

        self.wavelength = 532 * 10**(-9)
        self.npix = number_of_pixels
        self.f = 10 * 10**(-3)
        self.pixel_scale = math.sqrt(532*10**(-11)/number_of_pixels)
        self.r = 2.5 * 10**(-3)
        self.H = self.calc_phase_transmittance_freespace()
        self.H_lens = self.calc_phase_transmittance_freespace_lens()


    def calc_phase_transmittance_freespace_lens(self):
        k = np.pi * 2.0 / self.wavelength
        x, y = utils.mesh_grid(self.npix, self.pixel_scale)
        x, y = torch.tensor(x).to(device), torch.tensor(y).to(device)
        xy_squared = x ** 2 + y ** 2
        t1 = torch.exp(-(1.j * k) / (2 * self.f) * xy_squared)
        phi = torch.where(
            xy_squared <= self.r ** 2, t1, 1+0.j
        ).to(device)
        # TODO: maybe need to tensor entire function
        return phi

    def calc_phase_transmittance_freespace(self):
        x, y = utils.mesh_grid(self.npix, self.pixel_scale)
        x, y = torch.tensor(x).to(device), torch.tensor(y).to(device)
        f_x = (x / (self.pixel_scale ** 2 * self.npix))
        f_y = (y / (self.pixel_scale ** 2 * self.npix))
        k = np.pi * 2.0 / self.wavelength

        rhosqr = f_x ** 2 + f_y ** 2
        exp_1 = 1.j * k * self.f
        exp_2 = -1.j * np.pi * self.wavelength * self.f * rhosqr
        #TODO: may need to edit ks and xys to be tensor from the begining
        H = torch.exp(exp_1 + exp_2)
        return H

    def propagate_through_freespace(self, wavefront):
        wf_at_distance = utils.fft(wavefront)
        wf_at_distance = wf_at_distance * self.H
        wf_at_distance = utils.ifft(wf_at_distance)
        return wf_at_distance

    def convolution_4F(self, img, kernel):
        wavefront = img * torch.exp(1.j * torch.zeros(size=(self.npix,self.npix)).to(device))
        wavefront_1F = self.propagate_through_freespace(wavefront)
        wavefront_1Lens = wavefront_1F*self.H_lens
        wavefront_2F = self.propagate_through_freespace(wavefront_1Lens)
        wavefront_filtered = wavefront_2F * torch.fft.fftshift(torch.fft.fft2(kernel))
        wavefront_3F = self.propagate_through_freespace(wavefront_filtered)
        wavefront_2Lens = wavefront_3F*self.H_lens
        wavefront_4F = self.propagate_through_freespace(wavefront_2Lens)
        return torch.abs(wavefront_4F)

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
        return result

# if __name__ == '__main__':
#     img = io.imread("mnist-test.jpg", as_gray=True)
#     img = resize(img, (28,28),anti_aliasing=True)/255
#     plt.imshow(img, cmap='gray')
#     plt.show()
#     img1 = Variable(torch.tensor(img), requires_grad=True).to(device)
#     optics = Optics_simulation(img1.shape[0])
#     kernel = np.array(
#         [[1, 4, 7, 4, 1], [4, 16, 26, 16, 4], [7, 26, 41, 26, 7], [4, 16, 26, 16, 4], [1, 4, 7, 4, 1]])/26
#     kernel = Variable(torch.tensor(kernel, dtype=torch.float64), requires_grad=True).to(device)
#     output = optics.optConv2d(img1, kernel, True)
#     plt.imshow(output.cpu().detach().numpy(), cmap='gray')
#     plt.show()
#     plt.imshow(signal.correlate(img, kernel.cpu().detach().numpy(), mode="same"), cmap='gray')
#     plt.show()
