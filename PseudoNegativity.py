import numpy as np
import keras
from scipy import signal
import matplotlib.pyplot as plt
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
I = x_train[0]
k = np.array([[1,-2,3],[-4,5,-6],[7,-8,9]])
output = signal.convolve2d(I,k)
plt.imshow(output)
plt.show()

pos, neg = np.maximum(k,0) , np.maximum(k*(-1),0)
output_pos = signal.convolve2d(I,pos)
output_neg = signal.convolve2d(I,neg)
output_fin = output_pos-output_neg
plt.imshow(output_fin)
plt.imshow

comparison = np.sum(output_fin-output)
print(comparison)