from Optics_simulation import Optics_simulation
from matplotlib import pyplot as plt
import numpy as np
from scipy import signal
import scipy


INPUT_IMAGE = "textnontra.png"
sample_kernel = [[ 0.12813634,  0.11068335, 0,0],
                [ 0.26879886, -0,   0.00937528,0],
                [ 0.11395154, 0, -0,0],
                 [0,0,0,0]]


def kernel_in_frequency_domain(kernel):
    padded_kernel = np.pad(np.array(kernel), 148)
    kernel_fr = np.fft.fftshift(np.fft.fft2(padded_kernel))
    plt.imshow(abs(kernel_fr), cmap='gray')
    plt.title("Kernel FFT")
    plt.show()
    return kernel_fr

def fft_based_convolution():
    global sample_kernel
    padded_kernel = np.pad(np.array(sample_kernel), 148)
    img = plt.imread(INPUT_IMAGE)
    img = np.dot(img[..., :3], [0.299, 0.587, 0.114])
    img = np.pad(np.array(img),0)

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

    direct_convolution = scipy.signal.fftconvolve(img,sample_kernel)
    plt.imshow(abs(direct_convolution), cmap='gray')
    plt.title("direct convolution")
    plt.colorbar()
    plt.show()

if __name__=='__main__':
    fft_based_convolution()
    optics = Optics_simulation()
    output = optics.convolution_4F(INPUT_IMAGE, kernel_in_frequency_domain(sample_kernel))
    output = np.fft.fftshift(output)
    plt.imshow(output, cmap='gray')
    plt.show()
    # kernel_in_frequency_domain(sample_kernel)
    # plot_fft()
    # optics.no_convolution_4F(INPUT_IMAGE)