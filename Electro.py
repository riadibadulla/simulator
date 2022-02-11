import numpy as np
import scipy
from scipy.signal import convolve2d
from Optics_simulation import Optics_simulation

mat1 = np.array([[2,3,4]])
mat2 = np.array([[2,-1,2,4],[-5,2,1,3],[-5,2,1,2]])

def optical_convolution(matrix1, matrix2):
    simulator = Optics_simulation()

def conv_matmult(matrix1, matrix2, optical=False):
    results_size_x = matrix1.shape[0]
    results_size_y = matrix2.shape[1]
    size_of_padding = results_size_y-1
    for i in range(1,size_of_padding*(results_size_x-1)+results_size_x,size_of_padding+1):
        for j in range(size_of_padding):
            matrix1 = np.insert(matrix1, i, 0, axis=0)
    matrix1 = np.rot90(matrix1,1, axes=(0,-1))
    result = scipy.signal.fftconvolve(matrix1, matrix2)
    result = np.array([result[len(result) // 2]])
    result = result.reshape((results_size_x,results_size_y))
    return result

print(np.matmul(mat1,mat2))
print(conv_matmult(mat1,mat2))
