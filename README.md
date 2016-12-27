# Learning to learn and transfer knowledge with neural networks

This repository contains code that implements joint knowledge transfer and
learning of the training algorithm using neural networks. Python + Keras was used
to come up with this code.

The goal in creating this repository is to create general predictive models such that:

* Such models can be adapted to particular instance of supervised learning problem
without fine tuning and associated hyperparameter search and computational complexity;
* Such models can perform knowledge transfer from the known (training) instances of
 supervised learning problems to unknown (test) ones in order to adapt to unknown
 problems better.

One example applicaiton for such models could be to object segmentation or
tracking in video.

**This repository is work in progress.**

## General description

To implement joint learning with neural networks and knowledge transfer,
the following architecture is proposed:

<p align="center">
<img src="https://github.com/iaroslav-ai/learn-to-learn/blob/master/images/general_scheme.jpg?raw=true" alt="Generic optimization problem" height="280px" />
</p>

In this architecture, "learner" network takes as input a training dataset,
and produces outputs which either replace part of the weights of the "predictor"
network, or / and are fed into the predictor network as extra features.
In this scheme, learner is trained to configure predictor to be accurate,
and predictor network carries some prior knowledge captured during training
 about class of predictive modelling problems that allows predictor to be
 more accurate.

Such neural network architecture is trained using a gradient descent, using a dataset
of training and testing datasets.

## Two digit number classification example

Example task demonstrating proof of concept use of above architecture
is as follows: given one example of image of two digit number (based on MNIST)
detect whether an unlabeled image contains same number, eg see figure below:

<p align="center">
<img src="https://github.com/iaroslav-ai/learn-to-learn/blob/master/images/two_digit_example.jpg?raw=true" alt="Generic optimization problem" height="350px"/>
</p>

One can find the code implementing this in `mnist_example/main.py`. With
current unoptimized architecture the ~88% accuracy is achieved
(50% is best that is achieved by random guesses) on numbers that were not
used for training. This shows that such architecture can work, but likely
requires further parameter tuning. Note also that some digits are confusing
(eg 1 and 7) in MNIST, and as even one digit recognized wrongly causes
false negative, thus ideal error rate with two digit numbers is expected
to be higher that that of MNIST itself.
