# Author: Kevin Delassus - G00270791
# Adapted from: https://medium.com/coinmonks/handwritten-digit-prediction-using-convolutional-neural-networks-in-tensorflow-with-keras-and-live-5ebddf46dc8

# Imports
import keras as kr
from keras.datasets import mnist
import tkinter as tk
from PIL import ImageTk, Image

# load data from keras.datasets
(train_img, train_lbl), (test_img, y_test) = mnist.load_data()

# Reshape to the expected CNN format 
train_img = train_img.reshape(train_img.shape[0], train_img.shape[1], train_img.shape[2], 1).astype('float32')
test_img = test_img.reshape(test_img.shape[0], test_img.shape[1], test_img.shape[2], 1).astype('float32')

# One hot encode train_img & test_img
train_img/=255
test_img/=255

# one hot encode
train_lbl = kr.utils.to_categorical(train_lbl, 10)
y_test = kr.utils.to_categorical(y_test, 10)

#This creates the main window of an application
window = tk.Tk()
window.title("Join")
window.geometry("300x300")
window.configure(background='grey')

# Building a Convolutional
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


# Compiling of the Model
model.compile(loss='categorical_crossentropy', optimizer=kr.optimizers.Adam(), metrics=['accuracy'])

# Fit the model
model.fit(train_img, train_lbl, validation_data=(test_img, y_test), epochs=10, batch_size=200)

# Save the model for use within Jupyter Notebook
model.save('models/mnistModel.h5')

# Evaluation of the model
metrics = model.evaluate(test_img, y_test, verbose=0)
print("Metrics(Test loss & Test Accuracy): ")
print(metrics)
