from Optics_simulation import Optics_simulation
from matplotlib import pyplot as plt
import numpy as np
from scipy import signal
import scipy
from skimage import data
from skimage.color import rgb2gray
from skimage.transform import resize




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
        padded_kernel = np.pad(np.array(sample_kernel), padding_size)
        if padded_kernel.shape != img.shape:
            padded_kernel = np.pad(padded_kernel, ((0,1),(0,1)))
    else:
        padded_image = np.pad(np.array(img), padding_size)
        if padded_image.shape != sample_kernel.shape:
            padded_image = np.pad(padded_image, ((0,1),(0,1)))

    #fft of image
    signal_fr = np.fft.fftshift(np.fft.fft2(img))
    plt.imshow(np.log(abs(signal_fr)), cmap='gray')
    plt.title("Image FFT")
    plt.show()

    #kernel fft
    kernel_fr = np.fft.fftshift(np.fft.fft2(padded_kernel))
    plt.imshow(abs(kernel_fr), cmap='gray')
    plt.title("Kernel FFT")
    plt.show()


    result = np.fft.fftshift(np.fft.ifft2(np.multiply(signal_fr, kernel_fr)))
    plt.imshow(abs(result), cmap='gray')
    plt.colorbar()
    plt.title("FFTConvolved")
    plt.show()
    return abs(result)


def optConv2d(img,kernel,pseudo_negativity=True):
    if pseudo_negativity:
        pos, neg = np.maximum(kernel, 0), np.maximum(kernel * (-1), 0)
        output_pos = fft_based_convolution(INPUT_IMAGE, pos)
        output_neg = fft_based_convolution(INPUT_IMAGE, neg)
        output_fin = output_pos - output_neg
    else:
        output_fin = fft_based_convolution(INPUT_IMAGE, kernel)

    plt.imshow(output_fin)
    plt.show()

    plt.imshow(signal.convolve2d(img,kernel))
    plt.show()

if __name__=='__main__':
    kernel = np.random.random(size=(7, 7)) - 0.5
    INPUT_IMAGE = resize(rgb2gray(data.chelsea()), (450, 450)) / 255
    optConv2d(INPUT_IMAGE,kernel,True)
