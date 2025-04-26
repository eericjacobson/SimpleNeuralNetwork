import nntools as nnt
import numpy as np
import pandas as pd


def sigmoid(x):
    n = 1 + np.exp(-x)
    return 1.0/n

def d_sigmoid(x):
    sig = sigmoid(x)
    return sig * (1-sig)

def relu(x):
    return np.maximum(0, x)

def d_relu(x):
    return x > 0

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0, keepdims=True)

data = np.array(pd.read_csv("mnist/mnist_train.csv"))
actuals, input = data.T[0], data.T[1:]
layout_config = nnt.NetworkLayout([784,10,10], [sigmoid, softmax], [d_sigmoid])
nn = nnt.Network(layout=layout_config, input_layer=input, actuals=actuals)

nn.train(2, 100)



