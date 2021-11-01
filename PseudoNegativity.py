import numpy as np
import keras
from scipy import signal
import matplotlib.pyplot as plt

#load mnist dataset and take one image from it as an Input
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
I = x_train[0]

#using arbitrary values for 3x3 kernel
k = np.array([[1,-2,3],[-4,5,-6],[7,-8,9]])

#convolve the input and the kernel normally and ploy
output = signal.convolve2d(I,k)
plt.imshow(output)
plt.show()

#split kernel into negative and positive, convolve with the input seperately
# and subtract negative from positive
pos, neg = np.maximum(k,0) , np.maximum(k*(-1),0)
output_pos = signal.convolve2d(I,pos)
output_neg = signal.convolve2d(I,neg)
output_fin = output_pos-output_neg
plt.imshow(output_fin)
plt.imshow

#Check the distance between the resulting matrices
comparison = np.sum(output_fin-output)
print(comparison)