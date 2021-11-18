import numpy as np
import scipy
from scipy.signal import convolve2d

def conv_matmul(vector, matrix):
    vector = np.rot90(vector,1, axes=(0,1))
    result = convolve2d(vector,matrix)
    result = np.array([result[len(result)//2]])
    print(result)

# sample_vector = np.array([[2,3,4]])
# matrix = np.array([[5,3,5],[4,2,1],[1,5,6]])
# print(np.matmul(sample_vector,matrix))
# conv_matmul(sample_vector, matrix)
#

def matmult(matrix1,matrix2):
    results_size_x = matrix1.shape[0]
    results_size_y = matrix1.shape[1]
    matrix1 = np.rot90(matrix1,1, axes=(0,-1))
    matrix1 = np.insert(matrix1, 1, values=0, axis=1)
    result = convolve2d(matrix1, matrix2)
    result = np.array([result[len(result) // 2]])
    result = result.reshape((results_size_x,results_size_y))
    print(result)


mat1 = [[1,2],[1,4]]
mat2 = [[2,-1],[-5,2]]
mat1 = np.array(mat1)
mat2 = np.array(mat2)
matmult(mat1,mat2)
