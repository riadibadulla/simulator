import numpy as np
import scipy
from scipy.signal import convolve2d


# sample_vector = np.array([[2,3,4]])
# matrix = np.array([[5,3,5],[4,2,1],[1,5,6]])
#
# def conv_matmul(vector, matrix):
#     vector = np.rot90(vector,1, axes=(0,1))
#     result = scipy.signal.fftconvolve(vector,matrix)
#     result = np.array([result[len(result)//2]])
#     return result
#
#
# print(np.matmul(sample_vector,matrix))
# print(conv_matmul(sample_vector, matrix))


# mat1 = np.array([[1,2],[1,4]])
# mat2 = np.array([[2,-1],[-5,2]])

mat1 = np.array([[1,2,5],[1,4,5]])
mat2 = np.array([[2,-1,2],[-5,2,1],[-5,2,1]])

def conv_matmult(matrix1,matrix2):
    results_size_x = matrix1.shape[0]
    results_size_y = matrix2.shape[1]
    size_of_padding = results_size_y-results_size_x+1
    matrix1 = np.rot90(matrix1,1, axes=(0,-1))
    for i in range(size_of_padding):
        matrix1 = np.insert(matrix1, 1, values=0, axis=1)
    result = scipy.signal.fftconvolve(matrix1, matrix2)
    result = np.array([result[len(result) // 2]])
    result = result.reshape((results_size_x,results_size_y))
    return result

print(np.matmul(mat1,mat2))
print(conv_matmult(mat1,mat2))
