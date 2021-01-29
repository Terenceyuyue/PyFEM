#### Libraries
# Standard library
import random
import time
# Third-party libraries
import numpy as np

tic = time.time()

## Load MNIST
import mnist_loader

training_data_x, training_data_y, \
validation_data_x, validation_data_y, \
test_data_x, test_data_y = mnist_loader.load_data()

## Parameters
ndim = [784,15,10]
epochs = 10
mini_batch_size = 10
eta = 3

## Create a Network object
import network
nn = network.Network(ndim)

## Train network with SGD
nn.SGD(training_data_x, training_data_y, epochs, mini_batch_size, \
             eta, test_data_x, test_data_y)
    
## Recognize handwritten digits
num_p, yp, y = nn.evaluate(validation_data_x, validation_data_y)
ratio = num_p/len(y)
print("\n Recognize handwritten digits in validation_data \n")
print(" Accuracy = {:.2%} \n".format(ratio))

toc = (time.time() - tic);
print(' Elapsed time = %.4f s' % toc)