from Optics_simulation import Optics_simulation
from matplotlib import pyplot as plt
import numpy as np
from scipy import signal
import scipy
from skimage import io
from skimage.color import rgb2gray
from skimage.transform import resize
import seaborn as sns


if __name__=='__main__':
    img = io.imread("noisy.jpg")
    img = rgb2gray(img)
    optics = Optics_simulation(img.shape[0])
    kernel = np.array([[-1,-4,-7,-4,-1],[-4,16,26,16,-4],[-7,26,41,26,-7],[-4,16,26,16,-4],[-1,-4,-7,-4,-1]])
    sns.heatmap(kernel, annot=True)
    plt.show()
    output = optics.optConv2d(img,kernel, True)
    plt.imshow(output, cmap='gray')
    plt.show()
    plt.imshow(signal.fftconvolve(img, kernel, mode="same"), cmap='gray')
    plt.show()
