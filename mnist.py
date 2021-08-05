import numpy as np
import keras
from tensorflow.keras import layers
from tensorflow.python.client import device_lib
import matplotlib.pyplot as plt
from scipy import signal


# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")


# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(16, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(16, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

model.summary()

batch_size = 128
epochs = 1

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])


#get first layer kernels and reshape

kernels = model.weights[0][:,:,0,:].numpy().transpose(2,1,0)
print(kernels[0])

#draw one test input
print(x_test[0])
plt.imshow(x_test[0], cmap='Greys')
plt.colorbar()
plt.show()

#Draw first layer kernels

for kernel in kernels:
    plt.imshow(kernel, cmap='Greys')
    plt.colorbar()
    plt.show()


# get first layer outputs

model.pop()
model.pop()
model.pop()
model.pop()
model.pop()
model.pop()

#Get and draw first channel of first layer output

# first_layer_outputs = model.predict(np.array([x_test[0]]))
#
# one_output_channel = first_layer_outputs[0].transpose(2,0,1)[0]
#
# plt.imshow(one_output_channel)
# plt.show()
#
# print(one_output_channel)

#Perform convolutions manually

I = x_test[0].transpose(2,0,1)[0]
outputs = []
for kernel in kernels:
    outputs.append(signal.convolve2d(I,kernel))
outputs = np.array(outputs)

#tile kernels

z = 0
T = np.empty((12,12))

for i in range(0,10,3):
    for j in range(0,10,3):
        T[i:i+3,j:j+3] = kernels[z]
        z+=1
plt.imshow(T, cmap='Greys')
plt.show()


