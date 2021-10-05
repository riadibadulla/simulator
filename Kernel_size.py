import keras.activations
from matplotlib import pyplot
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import MaxPooling2D
from keras.layers import Flatten
batch_size = 1

(trainX, trainY), (testX, testY) = cifar10.load_data()

trainY = to_categorical(trainY)
testY = to_categorical(testY)

train_norm = trainX.astype('float32')
test_norm = testX.astype('float32')
# normalize to range 0-1
train_norm = train_norm / 255.0
test_norm = test_norm / 255.0


def define_model():
    model = Sequential()
    model.add(Conv2D(input_shape=(32,32,3),filters=16, kernel_size=(3,3), activation='elu', padding='same'))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='elu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='elu', padding='same'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='elu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='elu', padding='same'))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='elu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    return model

model = define_model()
opt = keras.optimizers.SGD(lr=0.001, momentum=0.9)
model.compile(optimizer=opt, loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()

model.fit(trainX,trainY,batch_size=20,epochs=5,validation_split=0.2)
model.evaluate(testX,testY,batch_size=5)