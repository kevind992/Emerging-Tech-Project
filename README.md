# README.md
## Software Requirements
To run the script and notebooks within this reposition I would recommend downloading and installing the following applications.
- [Anaconda](https://www.anaconda.com/). Anaconda contains most of the necessary tools to run the script and notebooks.
To download and install go to following [link](https://www.anaconda.com/download/) and download for your spesific system.  
- I would also recommend downloading and installing [Git](https://git-scm.com/) but this is not nessesary for running the script and notebooks. 

## Installing Tensorflow and Keras
Once Anaconda is installed we can now install Tensorflow and Keras. Type the following command to list the libraries which are already installed on your system.
```sh
   $ conda list
```
To install Tensorflow type
```sh
   $ conda install tensorflow
```
And for Keras type
```sh
   $ conda install keras
```
You should now be able to run the script and notebooks. If you get any errors it is more then likely that you might need to add another libraries. Make sure you read the error messages as they will indicate to you which libraries you need to install. 
## Cloning Repository
There is two ways to downloading the repository. If you have git installed run the following terminal command
```sh
   $ git clone https://github.com/kevind992/Emerging-Tech-Project.git
```
Just make sure that you are in the directory that your which clone the repository into. 
The second way is to download a zipped version of the repository. Once within the repository, click on the green **clone or download** button and press **Download ZIP**.
## Running Notebooks
Before you attempt to try and run the notebooks, create a directory called **data** within the Emerging-Tech-Project directory. Download the following link into the your newly created data directory

 - [train-images-idx3-ubyte.gz](http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz)
 - [train-labels-idx1-ubyte.gz](http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz)
 - [10k-images-idx3-ubyte.gz](http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz)
 - [t10k-labels-idx1-ubyte.gz](http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz)

To run the notebooks open a terminal and navigate to where you have cloned the repository. Once you are within the repository type the following 
```sh
   $ jupyter notebook
```
Jupyter Notebook will now open within a browser window. You can now click into the notebook of your choice. 
To run a specific notebook, press **Kernal** at the top of the window and then press **Restart & Run All**.
This will run all the python commands.
## Running the Script
To suc
