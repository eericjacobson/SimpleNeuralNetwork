import numpy as np
from matplotlib import pyplot as plt

class NetworkLayout():
    ACTIVATION = 0
    PRIME = 1

    layout = []
    functions = []

    def __init__(self, layout, afunctions, pfunctions) -> None:
        self.layout = layout
        self.functions = [afunctions, pfunctions]

class Network():
    layout : NetworkLayout
    weights = []
    biases = []
    input_layer = []
    actuals = []

    def __init__(self, layout : NetworkLayout, input_layer : np.matrix, actuals : np.matrix) -> None:
        self.layout = layout
        self.weights, self.biases = self.init_network(self.layout.layout)
        self.input_layer = input_layer
        self.actuals = actuals
    
    def train(self, learning_rate, iterations, debug=True):
        for i in range(iterations):
            layers, ulayers = self.fprop(self.weights, self.biases, self.layout.functions[NetworkLayout.ACTIVATION])
            weight_nudges, bias_nudges = self.bprop(layers, ulayers, self.weights, self.actuals, self.layout.functions[NetworkLayout.PRIME])
            self.weights, self.biases = self.nudge(self.weights, self.biases, weight_nudges, bias_nudges, learning_rate)
            if debug:
                if (i % 10 == 0):
                    print("Iterations: ", i)
                    print("Accuracy: ", self.get_accuracy(self.get_predictions(layers[-1]), self.actuals))


    def get_predictions(self, output_layer):
        return np.argmax(output_layer, 0)

    def get_accuracy(self, predictions, actuals):
        print(predictions, actuals)
        return np.sum(predictions == actuals) / actuals.size

    def one_hot(self, x):
        oh = np.zeros((x.size, x.max() + 1))
        oh[np.arange(x.size), x] = 1
        return oh.T

    def fprop(self, weights, biases, activation_functions):
        current_layer = self.input_layer
        ulayers = []
        layers = [self.input_layer]
        for i in range(len(activation_functions)):
            current_layer = weights[i].dot(current_layer) + biases[i]
            ulayers.append(current_layer)
            current_layer = activation_functions[i](current_layer)
            layers.append(current_layer)
        ulayers.pop()
        return layers, ulayers

    def bprop(self, layers, ulayers, weights, actuals, prime_functions):
        first_layer_loss = layers[-1] - self.one_hot(actuals)
        layer_deltas = self.calculate_layer_loss(first_layer_loss, ulayers, weights, prime_functions)
        weight_nudges, bias_nudges = self.calculate_nudges(layer_deltas, layers)
        return weight_nudges, bias_nudges

    def nudge(self, weights, biases, weight_nudges, bias_nudges, learning_rate):
        for i in range(len(weights)):
            weights[i] -= learning_rate * weight_nudges[i]
            biases[i] -= learning_rate * bias_nudges[i]
        return weights, biases
    
    def calculate_nudges(self, layer_deltas, layers):
        weight_nudges = []
        bias_nudges = []
        for i in range(len(layer_deltas)):
            sample_quotient = (1.0/layer_deltas[i].shape[1])
            weight_nudges.append(sample_quotient*layer_deltas[i].dot(layers[i].T))
            bias_nudges.append(sample_quotient*np.sum(layer_deltas[i], axis=1, keepdims=True))
        return weight_nudges, bias_nudges
            
    def calculate_layer_loss(self, first_layer_loss, ulayers, weights, prime_functions):
        layer_deltas = [first_layer_loss]
        for i in range(1, len(prime_functions)+1):
            layer_deltas.insert(0, weights[-i].T.dot(layer_deltas[-i]) * prime_functions[-i](ulayers[-i]))
        return layer_deltas
    
    def init_network(self, network_layout : list):
        weights, biases = [], []
        for i in range(len(network_layout)-1):
            weights.append(np.random.rand(network_layout[i+1], network_layout[i]) - 0.5)
            biases.append(np.random.rand(network_layout[i+1], 1) - 0.5)
        return weights, biases
