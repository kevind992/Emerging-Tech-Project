import numpy
from keras.datasets import mnist
import keras as kr
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.optimizers import Adam
from keras.utils import np_utils

# load data
(train_img, train_lbl), (test_img, y_test) = mnist.load_data()

# Reshaping to format which CNN expects (batch, height, width, channels)
train_img = train_img.reshape(train_img.shape[0], train_img.shape[1], train_img.shape[2], 1).astype('float32')
test_img = test_img.reshape(test_img.shape[0], test_img.shape[1], test_img.shape[2], 1).astype('float32')

# normalize inputs from 0-255 to 0-1
train_img/=255
test_img/=255

# one hot encode
train_lbl = np_utils.to_categorical(train_lbl, 10)
y_test = np_utils.to_categorical(y_test, 10)

# building a linear stack of layers with the sequential model
# Start a neural network, building it by layers.
# create model
model = kr.models.Sequential()

model.add(kr.layers.convolutional.Conv2D(32, (5, 5), input_shape=(train_img.shape[1], train_img.shape[2], 1), activation='relu'))
model.add(kr.layers.convolutional.MaxPooling2D(pool_size=(2, 2)))
model.add(kr.layers.convolutional.Conv2D(32, (3, 3), activation='relu'))
model.add(kr.layers.convolutional.MaxPooling2D(pool_size=(2, 2)))
model.add(kr.layers.Dropout(0.2))
model.add(kr.layers.Flatten())
model.add(kr.layers.Dense(128, activation='relu'))
model.add(kr.layers.Dense(10, activation='softmax'))


# Compile model
model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# Fit the model
model.fit(train_img, train_lbl, validation_data=(test_img, y_test), epochs=10, batch_size=200)

# Final evaluation of the model
metrics = model.evaluate(test_img, y_test, verbose=0)
print("Metrics(Test loss & Test Accuracy): ")
print(metrics)
