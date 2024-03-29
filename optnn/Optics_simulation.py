import math
import random

import numpy as np
from matplotlib import pyplot as plt
from torch.distributions import Poisson
from skimage import io
from skimage.transform import rescale, resize, downscale_local_mean
import seaborn as sns
from scipy import signal
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.modules.activation import ReLU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from . import utils
from matplotlib.colors import LinearSegmentedColormap
colors = [(0, 0, 0), (0, 1, 0)]
cmap_name = 'my_list'
cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=150)


class Optics_simulation:
    """A class which simulates the freespace optics

    :param number_of_pixels: size of the input. In case the kernel is larger than input, this should be the kernel size
    :type number_of_pixels: int
    """
    def __init__(self,number_of_pixels=28, full_padding_is_done=True):

        self.wavelength = 532 * 10**(-9)
        if not full_padding_is_done:
            self.npix = number_of_pixels+4
        else:
            self.npix = number_of_pixels
        self.full_padding_is_done = full_padding_is_done
        self.f = 10 * 10**(-3)
        self.pixel_scale = math.sqrt(532*10**(-11)/self.npix)
        self.r = 2.5 * 10**(-3)
        # self.H = torch.fft.fftshift(self.calc_phase_transmittance_freespace())
        self.H = torch.fft.ifftshift(self.calc_phase_transmittance_freespace())
        self.H_lens = self.calc_phase_transmittance_freespace_lens()

    def calc_phase_transmittance_freespace_lens(self):
        """Calculates the phase transittance of the lens

        :return: Phase transmittance of the convex lens
        :rtype: torch.Tensor
        """
        k = np.pi * 2.0 / self.wavelength
        x, y = utils.mesh_grid(self.npix, self.pixel_scale)
        x, y = torch.tensor(x), torch.tensor(y)
        xy_squared = x ** 2 + y ** 2
        t1 = torch.exp(-(1.j * k) / (2 * self.f) * xy_squared)
        phi = torch.where(
            xy_squared <= self.r ** 2, t1, 1+0.j
        )
        # TODO: maybe need to tensor entire function
        return phi

    def calc_phase_transmittance_freespace(self):
        """Calculates the phase transittance in freespace. h matrix

        :return: h matrix. Phase transmittance
        :rtype: torch.Tensor
        """
        x, y = utils.mesh_grid(self.npix, self.pixel_scale)
        x, y = torch.tensor(x), torch.tensor(y)
        f_x = (x / (self.pixel_scale ** 2 * self.npix))
        f_y = (y / (self.pixel_scale ** 2 * self.npix))
        k = np.pi * 2.0 / self.wavelength

        rhosqr = f_x ** 2 + f_y ** 2
        exp_1 = 1.j * k * self.f
        exp_2 = -1.j * np.pi * self.wavelength * self.f * rhosqr
        #TODO: may need to edit ks and xys to be tensor from the begining
        H = torch.exp(exp_1 + exp_2)
        return H

    def propagate_through_freespace(self, wavefront, device):
        """Propagates tge wavefront through freespace using angular spectrum method

        :param wavefront: wavefront in complex
        :type wavefront: torch.Tensor
        :return: wavefront at distance
        :rtype: torch.Tensor
        """
        wf_at_distance = utils.fft(wavefront)
        wf_at_distance = wf_at_distance * self.H.to(device)
        wf_at_distance = utils.ifft(wf_at_distance)
        return wf_at_distance

    def plot_wavefront(self, wavefront, vmax=None):
        """In case if need to plot the wavefront and save it

        :param wavefront: wavefront in complex
        :type wavefront: torch.Tensor
        :param vmax: vmax of the imshow
        :type vmax: float32
        """
        plt.imshow(torch.abs(wavefront).cpu().detach().numpy(), cmap=cmap, vmax=vmax)
        plt.axis("off")
        plt.savefig(str(random.randint(1,50))+"test.png", bbox_inches='tight')
        plt.show()

    def convolution_4F(self, img, kernel):
        """Performs the single convolution with optics

        :param img: input of the device(image)
        :type img: torch.Tensor
        :param kernel: Kernel of the convolution, already padded
        :type kernel: torch.Tensor
        :return: amplitude of the 4F system
        :rtype: torch.Tensor
        """
        device = img.device
        self.H = self.H.to(device)

        wavefront = img * torch.exp(1.j * torch.zeros(size=(self.npix,self.npix)).to(device))
        wavefront = self.propagate_through_freespace(wavefront, device)
        wavefront = wavefront*self.H_lens.to(device)
        wavefront = self.propagate_through_freespace(wavefront, device)
        wavefront = wavefront * torch.fft.fftshift(torch.fft.fft2(kernel))
        wavefront = self.propagate_through_freespace(wavefront, device)
        wavefront = wavefront*self.H_lens.to(device)
        wavefront = self.propagate_through_freespace(wavefront, device)
        return torch.abs(wavefront)

    def _apply_noise(self,output):
        output = output * 255
        noise_matrix = Poisson(output).sample()
        noise_matrix.requires_grad = False
        return (output + noise_matrix) / 255

    def optConv2d(self, img,kernel,pseudo_negativity=False,noise=True):
        """Performs the convolution, either with pseudo negativity or without, and fft shifts the output.

        :param img: input of the device(image)
        :type img: torch.Tensor
        :param kernel: Kernel of the convolution, already padded
        :type kernel: torch.Tensor
        :param pseudo_negativity: Apply Pseudo negativity if True
        :type pseudo_negativity: bool
        :return: convolved tensor
        :rtype: torch.Tensor
        """
        if not self.full_padding_is_done:
            img = torch.nn.functional.pad(img, (2, 2, 2, 2))
            kernel = torch.nn.functional.pad(kernel, (2, 2, 2, 2))
        if pseudo_negativity:
            relu = ReLU()
            pos, neg = relu(kernel), relu(kernel * (-1))

            output_pos = self.convolution_4F(img, pos)
            output_neg = self.convolution_4F(img, neg)
            if noise:
                output_pos = self._apply_noise(output_pos)
                output_neg = self._apply_noise(output_neg)
            result = torch.sub(output_pos,output_neg)
        else:
            result = self.convolution_4F(img, kernel)
        return torch.fft.fftshift(result)[...,4: self.npix, 4: self.npix]

if __name__ == '__main__':
    img = io.imread("mnist.jpg", as_gray=True)
    img = resize(img, (10,10),anti_aliasing=True)/255
    plt.imshow(img, cmap='gray')
    plt.axis("off")
    plt.savefig("intest.png", bbox_inches='tight')
    plt.show()
    img1 = Variable(torch.tensor(img), requires_grad=True).to(device)

    kernel = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    kernel = Variable(torch.tensor(kernel, dtype=torch.float64), requires_grad=True).to(device)
    # img1 = torch.nn.functional.pad(img1, (1, 1, 1, 1))
    kernel_padded = torch.nn.functional.pad(kernel, (3, 4, 3,4))
    optics = Optics_simulation(img1.shape[0],full_padding_is_done=False)
    output = optics.optConv2d(img1, kernel_padded, True)
    output = torch.rot90(torch.rot90(output))
    # [4: 14, 4: 14]
    plt.imshow(output.cpu().detach().numpy(), cmap='gray')
    plt.axis("off")
    plt.savefig("outtest.png", bbox_inches='tight')
    plt.show()
    # plt.imshow(torch.real(torch.fft.fftshift(torch.fft.ifft2(torch.fft.fft2(img1)*torch.fft.fft2(kernel_padded)))).cpu().detach().numpy()[:-2,:-2], cmap='gray')
    # plt.show()
    plt.imshow(signal.convolve(img, kernel.cpu().detach().numpy(), mode="same"), cmap='gray')
    plt.show()
