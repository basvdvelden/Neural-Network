import random as rnd
import numpy as np
from copy import deepcopy

LN = 0.1


def matrix(num_in, num_out):
    """
    Returns matrix with num_out vectors with num_in weights.
    :param num_in: amount of weights  (amount of inputs to layer)
    :param num_out: amount of neurons
    :rtype: list[list[int]]
    """
    matrix = []
    for n in range(num_out):
        matrix.append([])
        for m in range(num_in):
            rand = rnd.random()
            matrix[n].append(rand)
    return matrix


def sigmoid(activation):
    """
    Returns sigmoid(activation).
    :param activation: number to squash
    :rtype: float
    """
    e = np.exp(-activation)
    result = 1 / (1 + e)
    return result


class NeuralNetwork(object):

    def __init__(self, num_in, num_out, num_hls):
        """        
        :param num_in: amount of input neurons
        :param num_out: amount of output neurons
        :param num_hls: amount of hidden layers
        """
        self.output_layer = Layer(num_in, num_out)
        self.hidden_layers = []
        for num_hl in range(num_hls):
            self.hidden_layers.append(Layer(num_in, num_in))

    def feed_forward(self, inputs):
        """
        Feed inputs forward to layers.
        :type inputs: list[int, float]
        :return: outputs
        """
        for hl in self.hidden_layers:
            inputs = hl.feed_forward(inputs)
        return self.output_layer.feed_forward(inputs)

    def train(self, inputs, targets):
        """
        First feeds inputs forward, then computes errors for all units, then we update the weights.
        :param inputs: 1 set of inputs
        :param targets: 1 set of targets for given inputs
        :type inputs: list
        :type targets: list
        """
        self.feed_forward(inputs)

        output_errors = np.array([], dtype=float)

        for ind, output_neuron in enumerate(self.output_layer.neurons):
            error = output_neuron.get_output_layer_error(targets[ind])
            output_errors = np.insert(output_errors, ind, error)
        if not self.hidden_layers:
            for ind, neuron in enumerate(self.output_layer.neurons):
                neuron.update_weights(output_errors[ind])
            return
        
        hidden_layers = deepcopy(self.hidden_layers)
        hl_errors = [[]]
        hidden_layer = hidden_layers[len(self.hidden_layers) - 1]

        for ind, hl_neuron in enumerate(hidden_layer.neurons):

            weights = []

            for output_neuron in self.output_layer.neurons:
                weights.append(output_neuron.weights[ind])

            error = hl_neuron.get_hidden_layer_error(output_errors, weights)
            hl_errors[0].append(error)

        hidden_layers = list(reversed(hidden_layers))[:1]
        next_hl = hidden_layer

        for hl_ind, hidden_layer in enumerate(hidden_layers):
            hl_errors.append([])
            for neuron_ind, neuron in enumerate(hidden_layer.neurons):

                weights = []

                for next_neuron in next_hl.neurons:
                    weights.append(next_neuron.weights[neuron_ind])

                error = neuron.get_hidden_layer_error(hl_errors[hl_ind], weights)
                hl_errors[hl_ind + 1].append(error)
            next_hl = hidden_layer

        self.update_weights(output_errors, hl_errors, hidden_layers)
        return True

    def update_weights(self, out_errors, hl_errors, hidden_layers):
        """
        Update all weights in the network.
        :param out_errors: array of output layer errors
        :param hl_errors: array of arrays of hidden layer errors
        :param hidden_layers: hidden layers to update weights
        """
        self.output_layer.update_bias(sum(out_errors))
        
        for output_error, neuron in zip(out_errors, self.output_layer.neurons):
            neuron.update_weights(output_error)

        for ind, hl in enumerate(hidden_layers):
            hl.update_bias(sum(hl_errors[ind]))
            for neuron in hl.neurons:
                for hl_error in hl_errors[ind]:
                    neuron.update_weights(hl_error)


class Layer(object):
    def __init__(self, num_in, num_out, bias=None):
        """
        :param num_in: amount of inputs per neuron of this layer
        :param num_out: amount of neurons of this layer
        :param bias: default is 1
        """
        self.bias = bias if bias else 1
        self.matrix = np.array(matrix(num_in, num_out))
        self.neurons = []

        for num_neuron in range(num_out):
            self.neurons.append(Neuron(self.matrix[num_neuron], self.bias))

        self.inputs = None

    def feed_forward(self, inputs):
        """
        Feed inputs forward to neurons.
        :type inputs: list[int, float]
        :return: outputs
        """
        self.inputs = inputs
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.output(inputs))
        return outputs

    def update_bias(self, error):
        """
        Updates bias of this layer.
        :param error: sum of errors of this layer
        """
        gradient = error * LN
        self.bias -= gradient


class Neuron(object):
    def __init__(self, weights, bias):
        """
        :param weights: array of weights
        :param bias: bias of this neuron's layer
        """
        self.bias = bias
        self.weights = weights
        self.out = None
        self.inputs = None

    def weighted_sum(self, inputs):
        """
        Returns dot product of weights and given inputs + bias.
        :param inputs: array of inputs
        :rtype: float
        """
        return np.dot(inputs, self.weights) + self.bias

    def output(self, inputs):
        """
        Returns activation.
        :param inputs: neuron input
        """
        result = sigmoid(self.weighted_sum(inputs))
        self.out = result
        self.inputs = inputs
        return result

    def get_output_layer_error(self, target):
        """
        Return error of output layer, only called if this neuron is in output layer.
        :param target: target output
        :return: 2 * (self.out - target) * (1 - self.out) * self.out
        """
        return 2 * (self.out - target) * (1 - self.out) * self.out

    def get_hidden_layer_error(self, prev_errors, weights):
        """
        Returns error of this neuron. Only called if this neuron is in a hidden layer
        :param prev_errors: errors of 1 layer in front of this layer
        :param weights: weights of 1 layer in front of this layer
        :return: self.out * (1 - self.out) * np.dot(prev_errors, weights) * 2
        """
        return self.out * (1 - self.out) * np.dot(prev_errors, weights) * 2

    def update_weights(self, error):
        """
        Update all weights of this neuron.
        :param error: this neuron's error
        """
        for ind, weight in enumerate(self.weights):
            gradient = error * self.inputs[ind] * LN
            weight -= gradient
            self.weights[ind] = weight
