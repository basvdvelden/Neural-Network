
import math
import random


LN = 0.3


def matrix(numIN, numON):
    matrix = []
    for i in range(numON):
        matrix.append([])
        for c in range(numIN):
            rw = random.random()
            matrix[i].append(rw)
    return matrix


class NeuralNetwork:

    def __init__(self, numIN, numON, numHL):
        self.hidden_layers = []
        for i in range(numHL):
            self.hidden_layers.append(HiddenLayer(numIN))

        self.output_layer = OutputLayer(numIN, numON)

    def train(self, t_inputs, t_targets):
        self.feed_forward(t_inputs)
        ol_deltas = []
        for n in range(len(self.output_layer.neurons)):
            ol_deltas.append(self.output_layer.neurons[n].ol_delta(t_targets, n))
        hls_deltas = []
        hls_delta_i = 0
        for hl in range(len(self.hidden_layers), -1, -1):
            hl -= 1
            if hl != -1:
                pd_E_wrt_hn_tot_net_input = [0] * len(self.hidden_layers[hl].neurons)

                for h in range(len(self.hidden_layers[hl].neurons)):
                    d_E_wrt_hno = 0
                    if hl == len(self.hidden_layers) - 1:
                        for o in range(len(self.output_layer.neurons)):
                            d_E_wrt_hno += ol_deltas[o] * self.output_layer.neurons[o].weights[h]
                        pd_E_wrt_hn_tot_net_input[h] = d_E_wrt_hno * self.hidden_layers[hl].neurons[h].pd_total_net_wrt_input()
                    else:
                        n_hl = hl + 1
                        for o in range(len(self.hidden_layers[n_hl].neurons)):
                            d_E_wrt_hno += hls_deltas[hls_delta_i][o] * self.hidden_layers[n_hl].neurons[o].weights[h]

                        pd_E_wrt_hn_tot_net_input[h] = d_E_wrt_hno * self.hidden_layers[hl].neurons[h].pd_total_net_wrt_input()

                hls_deltas.append(pd_E_wrt_hn_tot_net_input)
                if hl != len(self.hidden_layers) - 1:
                    hls_delta_i += 1
        for o in range(len(self.output_layer.neurons)):
            for w in range(len(self.output_layer.neurons[o].weights)):
                pd_E_wrt_weight = ol_deltas[o] * self.output_layer.neurons[o].inputs[w]
                self.output_layer.neurons[o].weights[w] -= LN * pd_E_wrt_weight
        
        hl_i = len(self.hidden_layers) - 1
        for hl_d in range(len(hls_deltas)):
            for n in range(len(self.hidden_layers[hl_i].neurons)):
                for w in range(len(self.hidden_layers[hl_i].neurons[n].weights)):
                    pd_E_wrt_weight = hls_deltas[hl_d][n] * self.hidden_layers[hl_i].neurons[n].inputs[w]
                    self.hidden_layers[hl_i].neurons[n].weights[w] -= LN * pd_E_wrt_weight
            hl_i -= 1

    def feed_forward(self, inputs):
        for i in range(len(self.hidden_layers)):
            if i == 0:
                ifo = self.hidden_layers[i].feed_forward(inputs)
            else:
                ifo = self.hidden_layers[i].feed_forward(ifo)

        return self.output_layer.feed_forward(ifo)


class HiddenLayer:

    def __init__(self, numIN):

        self.bias = 1
        self.neurons = []
        self.matrix = matrix(numIN, numIN)
        for o in range(numIN):
            self.neurons.append(Neuron(self.matrix, o))

    def feed_forward(self, inputs):
        outputs = []
        for i in self.neurons:
            outputs.append(i.activation(inputs, self.bias))
        return outputs


class OutputLayer:

    def __init__(self, numIN, numON):

        self.bias = 1
        self.matrix = matrix(numIN, numON)
        self.neurons = []

        for o in range(numON):
            self.neurons.append(Neuron(self.matrix, o))

    def feed_forward(self, inputs):
        outputs = []
        for i in self.neurons:
            outputs.append(i.activation(inputs, self.bias))
        return outputs


class Neuron:
    def __init__(self, matrix, index_of_M):

        self.weights = matrix[index_of_M]
        self.output = None

    def weighted_sum(self, weights, inputs, bias):
        ind = 0
        weighted_input = []
        for i in weights:
            output = i * inputs[ind]
            weighted_input.append(output)
            ind += 1

        list = sum(weighted_input) + bias
        return list

    def sigmoid(self, prediction):
        e = math.exp(-prediction)
        prediction = 1 / (1 + e)
        return round(prediction, 8)

    def pd_total_net_wrt_input(self):
        return self.output * (1 - self.output)

    def activation(self, inputs, bias):
        self.inputs = inputs
        output = self.sigmoid(self.weighted_sum(self.weights, inputs, bias))
        self.output = output
        return output

    def ol_delta(self, targets, index):
        ol_delta = (self.output - targets[index]) * self.output * (1 - self.output)
        return ol_delta

# NeuralNetwork(num_of_inputs, num_of_outputs, num_of_hidden_layers)
# nn = NeuralNetwork(2, 2, 3)
# inputs = [[10, 25], [20, 50], [12, 30], [6, 15], [4, 16], [6, 24], [5, 20], [3, 12]]
# targets = [[1, 0], [1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1], [0, 1]]
