from Optics_simulation import Optics_simulation
from matplotlib import pyplot as plt
import numpy as np
from scipy import signal
import scipy




INPUT_IMAGE = "textnontra.png"
sample_kernel = [[ 0.12813634,  0.11068335, -0.14977527,0],
                [ 0.26879886, -0.0361914,   0.00937528,0],
                [ 0.11395154, -0.12251621, -0.3273148,0],
                 [0,0,0,0]]


def kernel_in_frequency_domain(kernel):
    padded_kernel = np.pad(np.array(kernel), 148)
    dark_image_grey_fourier = np.fft.fftshift(np.fft.fft2(padded_kernel))
    print(dark_image_grey_fourier)
    dark_image_grey_fourier = np.absolute(dark_image_grey_fourier)
    plt.imshow(dark_image_grey_fourier)
    plt.show()
    print(dark_image_grey_fourier)
    return dark_image_grey_fourier

def plot_fft():
    global sample_kernel
    img = plt.imread(INPUT_IMAGE)
    img = np.dot(img[..., :3], [0.299, 0.587, 0.114])
    sample_kernel = np.pad(np.array(sample_kernel), 148)
    dark_image_grey_fourier = np.fft.fftshift(np.fft.fft2(sample_kernel))
    plt.figure(num=None, figsize=(5, 5), dpi=130)
    plt.imshow(np.log(abs(dark_image_grey_fourier)), cmap='gray')
    plt.show()

def fft_based_convolution():
    global sample_kernel
    padded_kernel = np.pad(np.array(sample_kernel), 148)
    img = plt.imread(INPUT_IMAGE)
    img = np.dot(img[..., :3], [0.299, 0.587, 0.114])
    img = np.pad(np.array(img),0)
    plt.imshow(img, cmap="Greys")
    plt.show()
    signal_fr = np.fft.rfft2(img)
    print(type(signal_fr))
    print(signal_fr.shape)
    # plt.imshow(np.imag(np.fft.rfft2(img, axes=(-1, -1))))
    # plt.show()
    plt.imshow(np.imag(signal_fr), cmap="Greys")
    plt.show()
    kernel_fr = np.fft.rfft2(padded_kernel)
    plt.imshow(np.imag(kernel_fr), cmap="Greys")
    plt.show()
    result = np.fft.irfft2(np.multiply(signal_fr, kernel_fr))
    # np.fft.ifftshift(
    plt.imshow(np.fft.ifftshift(result), cmap="Greys")
    plt.show()

    direct_convolution = scipy.signal.fftconvolve(img,sample_kernel)
    plt.imshow(direct_convolution, cmap="Greys")
    plt.title("direct convolution")
    plt.show()

if __name__=='__main__':
    # optics = Optics_simulation()
    # optics.convolution_4F(INPUT_IMAGE, kernel_in_frequency_domain(sample_kernel))
    # kernel_in_frequency_domain(sample_kernel)
    fft_based_convolution()
    # plot_fft()
    # optics.no_convolution_4F(INPUT_IMAGE)