import math
import random

import numpy as np
from matplotlib import pyplot as plt
from skimage import io
from skimage.transform import rescale, resize, downscale_local_mean
import seaborn as sns
from scipy import signal
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.modules.activation import ReLU
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import utils
from matplotlib.colors import LinearSegmentedColormap
colors = [(0, 0, 0), (0, 1, 0)]
cmap_name = 'my_list'
cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=150)


class Optics_simulation:
    """A class which simulates the freespace optics

    :param number_of_pixels: size of the input. In case the kernel is larger than input, this should be the kernel size
    :type number_of_pixels: int
    """
    def __init__(self,number_of_pixels=28):

        self.wavelength = 532 * 10**(-9)
        self.npix = number_of_pixels
        self.f = 10 * 10**(-3)
        self.pixel_scale = math.sqrt(532*10**(-11)/number_of_pixels)
        self.r = 2.5 * 10**(-3)
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

    def propagate_through_freespace(self, wavefront):
        """Propagates tge wavefront through freespace using angular spectrum method

        :param wavefront: wavefront in complex
        :type wavefront: torch.Tensor
        :return: wavefront at distance
        :rtype: torch.Tensor
        """
        wf_at_distance = utils.fft(wavefront)
        wf_at_distance = wf_at_distance * self.H
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
        self.H_lens = self.H_lens.to(device)

        wavefront = img * torch.exp(1.j * torch.zeros(size=(self.npix,self.npix)).to(device))
        wavefront = self.propagate_through_freespace(wavefront)
        wavefront = wavefront*self.H_lens
        wavefront = self.propagate_through_freespace(wavefront)
        wavefront = wavefront * torch.fft.fftshift(torch.fft.fft2(kernel))
        wavefront = self.propagate_through_freespace(wavefront)
        wavefront = wavefront*self.H_lens
        wavefront = self.propagate_through_freespace(wavefront)

        # self.plot_wavefront(wavefront)
        # self.plot_wavefront(wavefront_1F)
        # self.plot_wavefront(wavefront_1Lens)
        # self.plot_wavefront(wavefront_2F, vmax=0.0025)
        # self.plot_wavefront(wavefront_filtered, vmax=0.0025)
        # self.plot_wavefront(torch.fft.ifftshift(wavefront_3F))
        # self.plot_wavefront(torch.fft.ifftshift(wavefront_2Lens))
        # self.plot_wavefront(torch.fft.ifftshift(wavefront_4F))
        # self.plot_wavefront(torch.fft.fftshift(torch.fft.fft2(kernel)))
        return torch.abs(wavefront)

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
        """Performs the convolution, either with pseudo negativity or wigthout, and fft shifts the output.

        :param img: input of the device(image)
        :type img: torch.Tensor
        :param kernel: Kernel of the convolution, already padded
        :type kernel: torch.Tensor
        :param pseudo_negativity: Apply Pseudo negativity if True
        :type pseudo_negativity: bool
        :return: convolved tensor
        :rtype: torch.Tensor
        """
        # img, kernel = self.process_inputs(img, kernel)
        if pseudo_negativity:
            relu = ReLU()
            pos, neg = relu(kernel), relu(kernel * (-1))

            output_pos = self.convolution_4F(img, pos)
            output_neg = self.convolution_4F(img, neg)
            # output_pos = torch.fft.ifft(torch.fft.fft2(img)*torch.fft.fft2(pos))
            # output_neg = torch.fft.ifft(torch.fft.fft2(img)*torch.fft.fft2(neg))
            result = torch.sub(output_neg,output_pos)
        else:
            result = self.convolution_4F(img, kernel)
        result = torch.fft.fftshift(result)
        return result

if __name__ == '__main__':
    img = io.imread("mnist-test.jpg", as_gray=True)
    img = resize(img, (28,28),anti_aliasing=True)/255
    plt.imshow(img, cmap='gray')
    plt.axis("off")
    plt.savefig("intest.png", bbox_inches='tight')
    plt.show()
    img1 = Variable(torch.tensor(img), requires_grad=True).to(device)
    optics = Optics_simulation(img1.shape[0])
    kernel = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    # kernel = np.array(
    #     [[-1, -1, -1, -1, -1], [-1, -1, -1, -1, -1], [-1, -1, 9, -1, -1], [-1, -1, -1, -1, -1], [-1, -1, -1, -1, -1]])
    sns.heatmap(kernel)
    plt.show()
    kernel = Variable(torch.tensor(kernel, dtype=torch.float64), requires_grad=True).to(device)
    output = optics.optConv2d(img1, kernel, True)
    output = torch.rot90(torch.rot90(output))
    plt.imshow(output.cpu().detach().numpy(), cmap='gray')
    plt.axis("off")
    plt.savefig("outtest.png", bbox_inches='tight')
    plt.show()
    plt.imshow(signal.correlate(img, kernel.cpu().detach().numpy(), mode="same"), cmap='gray')
    plt.show()
