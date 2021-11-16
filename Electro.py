import numpy as np
import scipy
from scipy.signal import convolve2d

def conv_matmul(vector, matrix):
    vector = np.rot90(vector,1, axes=(0,1))
    result = convolve2d(vector,matrix)
    result = np.array([result[len(result)//2]])
    print(result)

sample_vector = np.array([[2,3,4]])
matrix = np.array([[5,3,5],[4,2,1],[1,5,6]])
print(np.matmul(sample_vector,matrix))
conv_matmul(sample_vector, matrix)