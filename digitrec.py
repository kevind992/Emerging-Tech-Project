# Author: Kevin Delassus - G00270791
# Adapted from: https://medium.com/coinmonks/handwritten-digit-prediction-using-convolutional-neural-networks-in-tensorflow-with-keras-and-live-5ebddf46dc8
#               https://pythonspot.com/tk-file-dialogs/
#               https://towardsdatascience.com/basics-of-image-classification-with-keras-43779a299c8b

# Imports
import keras as kr
from keras.datasets import mnist
from tkinter import filedialog
from tkinter import *
import numpy as np
import cv2
import matplotlib.pyplot as plt

def user_Input() :

    print()
    print("Please Select from the following Options:")
    print("============================================")
    print("Enter '1' to Train a new Nerual Network")
    print("Enter '2' to make a Prediction using an exsisting model")
    print("Enter '0' to Exit")
    inputNum = input()
    print("You have entered " + inputNum)
    print(type(inputNum))

   # while inputNum!=0:
    if inputNum == '1':
        epochs = int(input("Please Define amount of Epochs: "))
        trainNetwork(epochs)
    elif inputNum == '2':
        testNetwork()
    else:
        print("Invalid Option.")

def trainNetwork(epochsNum):

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
    model.fit(train_img, train_lbl, validation_data=(test_img, y_test), epochs=epochsNum, batch_size=200)

    # Save the model for use within Jupyter Notebook
    model.save('models/mnistModel.h5')
    print("Model saved to 'models/mnistModel.h5'..")

    # Evaluation of the model
    metrics = model.evaluate(test_img, y_test, verbose=0)
    print("Metrics(Test loss & Test Accuracy): ")
    print(metrics)

def testNetwork():

    # Asking user to enter own image for testing
    root = Tk()
    root.testImage =  filedialog.askopenfilename(initialdir = "C:\\",title = "Select Image",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
    print(root.testImage)

    # Reading image and resizing it to correct CNN format
    imgFile = cv2.imread(root.testImage)
    img = cv2.resize(imgFile, (28, 28))
    arr = img.reshape(-1,28, 28, 1).astype('float32')

    # One hot encode arr
    arr/=255
    # Making prediction
    result = model.predict_classes(arr)

    prediction = result[0]

    # Displaying prediction
    print("Class: ",prediction)

    # Showing image and predicted result
    plt.imshow(imgFile)
    plt.title(prediction)
    plt.show()


def main() :
    user_Input()

if __name__ == '__main__':
    print("Running")
    main()
