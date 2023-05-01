# FCNN-NUMPY

## Introduction
A modular implemenataion of FCNN using only numpy.
I implemented this even though there are many other libraries available because I wanted to understand the working of neural networks better.

## Design
The neural network was designed to be modular for easy customization, allowing for the addition of different types and quantities of layers, changing optimizers, and other modifications with ease.
## Layers Creation
To achieve this, a layer was created as an abstract module, with linear, ReLU, and combined layers being implemented. Each layer could have varying numbers of input and output neurons, with each layer having a forward and backward step. The network calls these layers in loops to perform the steps, without the network being aware of the specifics of each layer.

In building the model, a list of hidden layer sizes is passed to the network, allowing for the creation of any desired network size. Batch stochastic gradient descent (SGD) is used, with the batch size being adjustable via the network's parameter.
## Learning Rate
The network accepts a learning rate and a decay for the learning rate, with the learning rate decreasing after each epoch for greater accuracy. 
## Optimizers
An abstract optimizer was implemented in a modular way to allow for easy switching between different optimizers, with three different optimizers being created: SGD, SGD with momentum, and Adam. After testing, Adam was selected for use, with a learning rate of 5e-3, learning rate decay of 0.99, beta 1 of 0.9, and beta 2 of 0.999.
## Weight Initialization
In testing, weight initialization was found to have a significant effect on results. Various methods were tested, including normal distribution and uniform distribution between 1 and -1, as well as Xavier Initialization and Normalized Xavier Initialization, which were effective but had issues with ReLU. He Weight Initialization, which is adapted to ReLU, was ultimately used, with significant improvement in results.
## Results
Using these parameters, the network was able to achieve 100% accuracy on the training sample and over 98% on the validation sample in just a few minutes.
