import astropy.units as u
import numpy as np
import pyoptica as po
from matplotlib import pyplot as plt

wavelength = 500 * u.nm
pixel_scale = 0.1 * u.um
npix = 300
na = 0.35
coherence_factor = 0
f= 0.0006 * u.cm
axis_unit = u.mm
wf = po.Wavefront(wavelength, pixel_scale, npix)
r = 2 * u.mm
lens = po.ThinLens(2*r, f)
fs = po.FreeSpace(f)


def rgb2gray(rgb):
    img = np.dot(rgb[..., :3], [0.299, 0.587, 0.114])
    return img

def modulate_signal(image_dir):
    img = plt.imread(image_dir)
    target = rgb2gray(img)
    wf.amplitude = target

def plot_wavefront(wavefront, amplitude):
    fig = wavefront.plot(amplitude=amplitude, fig_options=dict(figsize=(5, 5), dpi=130))
    fig[0].show()

def make_4F_engine(waveform):
    return waveform * fs * lens * fs * fs * lens * fs

def make_fourier_engine(waveform):
    return waveform * fs * lens * fs

def no_convolution_4F():
    modulate_signal("textnontra.png")
    plot_wavefront(wf, dict(vmax=1))
    # print(fs.calc_optimal_distance(wf))
    wf_imaged = make_4F_engine(wf)
    print(type(wf_imaged))
    plot_wavefront(wf_imaged, dict(vmax=1))

def convolution_4F():
    modulate_signal("textnontra.png")
    plot_wavefront(wf, dict(vmax=1))
    wf_in_frequency_domain = make_fourier_engine(wf)
    plot_wavefront(wf_in_frequency_domain, dict(vmax=1))

    amplitude_in_fourier = wf_in_frequency_domain.amplitude
    kernel = np.pad(np.array([[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]]), 148)
    convoluted_matrix = amplitude_in_fourier * kernel
    wf.amplitude = convoluted_matrix
    wf_imaged = make_fourier_engine(wf)
    plot_wavefront(wf_imaged, 'default')

no_convolution_4F()
convolution_4F()

