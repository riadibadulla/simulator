from Optics_simulation import Optics_simulation
from matplotlib import pyplot as plt
import numpy as np
from scipy import signal
import scipy
from skimage import io
from skimage.color import rgb2gray
from skimage.transform import resize
import seaborn as sns

def kernel_in_frequency_domain(kernel):
    padded_kernel = np.pad(np.array(kernel), 223)
    kernel_fr = np.fft.fftshift(np.fft.fft2(padded_kernel))
    # plt.imshow(abs(kernel_fr), cmap='gray')
    # plt.title("Kernel FFT")
    # plt.show()
    return kernel_fr

def fft_based_convolution(img, sample_kernel):

    size_of_image = img.shape[0]
    size_of_kernel = sample_kernel.shape[0]
    padding_size = abs(size_of_image - size_of_kernel) // 2

    if size_of_image>size_of_kernel:
        # img = np.pad(img, (size_of_kernel-1)//2)
        # size_of_image = img.shape[0]
        # print(size_of_image)
        # padding_size = abs(size_of_image - size_of_kernel) // 2
        padded_kernel = np.pad(np.array(sample_kernel), padding_size)
        print(padded_kernel.shape[0])
        if padded_kernel.shape != img.shape:
            padded_kernel = np.pad(padded_kernel, ((0,1),(0,1)))
    else:
        padded_image = np.pad(np.array(img), padding_size)
        if padded_image.shape != sample_kernel.shape:
            padded_image = np.pad(padded_image, ((0,1),(0,1)))

    #kernel fft
    kernel_fr = np.fft.fftshift(np.fft.fft2(padded_kernel))
    optics = Optics_simulation(size_of_image)
    result = optics.convolution_4F(img,kernel_fr)
    result = np.fft.fftshift(result)
    return result


def optConv2d(img,kernel,pseudo_negativity=True):
    optics = Optics_simulation(img.shape[0])
    if pseudo_negativity:
        pos, neg = np.maximum(kernel, 0), np.maximum(kernel * (-1), 0)
        output_pos = fft_based_convolution(INPUT_IMAGE, pos)
        output_pos = optics.no_convolution_4F(output_pos)
        output_neg = fft_based_convolution(INPUT_IMAGE, neg)
        optics = Optics_simulation(img.shape[0])
        output_neg = optics.no_convolution_4F(output_neg)
        output_fin = output_pos - output_neg
    else:
        output_fin = fft_based_convolution(img, kernel)
        output_fin = optics.no_convolution_4F(output_fin)

    plt.imshow(output_fin, cmap='gray')
    plt.show()
    plt.imshow(signal.fftconvolve(img,kernel, mode="same"), cmap='gray')
    plt.show()

if __name__=='__main__':
    img = io.imread("noisy.jpg")
    img = rgb2gray(img)
    kernel = np.array([[1,4,7,4,1],[4,16,26,16,4],[7,26,41,26,7],[4,16,26,16,4],[1,4,7,4,1]])
    sns.heatmap(kernel, annot=True)
    plt.show()

    optConv2d(img,kernel,False)
