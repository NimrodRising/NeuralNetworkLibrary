import numpy as np

class NeuralNetwork:
    def __init__(self, shape):
        self.shape = shape
        self.weights, self.biases = self.xavier(shape)
        self.input = np.random.normal(loc = 0.0, scale=1.0, size = (shape[0], 1))
        self.activations = []

    @staticmethod
    def xavier(shape):
        weights = []
        biases = []
        for i in range(len(shape) - 1):
            rows = shape[i + 1]
            cols = shape[i]
            bias_vector = np.zeros((rows, 1))
            biases.append(bias_vector)
            weight_matrix = np.matrix(np.random.normal(loc=0.0, scale=1.0, size = (rows, cols)))*np.sqrt(2/rows)
            weights.append(weight_matrix)
        return (weights, biases)
    
    def feedforward(self):
        activation = self.input
        self.activations.append(activation)
        for i in range(len(self.shape) - 1):
            bias = self.biases[i]
            weight_matrix = self.weights[i]
            next_activation = np.asarray(self.ReLU(weight_matrix*activation + bias))
            activation = next_activation  
            self.activations.append(activation)
        output = activation
        return output
    
    @staticmethod
    def ReLU(x):
        return np.maximum(0, x)
    
    # training_data is an array consisting of all the training data, dimension of el of data should equal dimension of input layer
    # labels is an array consisting of all the correct labels
    # default batch size is 1 in which case gradient descent is stochastic
    def train(self, training_data, labels, batch_size = 1):
        for item in training_data:
            output = self.feedforward(item)
            self.weights, self.biases = self.backpropagate(output)
    
    @staticmethod
    def backpropagate(item, output):
        # find all deltas
        delta = 2*(item - output)*output*(1 - item)
        # calculate gradient matrix for weights and biases
        return (new_weights, new_biases)

    @staticmethod
    # Delta L, j = 2*(item - output)*output*(1 - item), can be done vector wise
    # Delta l, j = sum over neuroons in l: delta (l+1) * weight (l+1) * activattion (l) * (1 - activation (l))

    def delta(layer, neuron, item, output):
        weights = NeuralNetwork.weights
        layers_left = layer - 1
        if (layers_left == 0):
            delta = 2*(item - output)*output*(1 - item)
            delta_j = delta[neuron]
        else:
            activation = NeuralNetwork.activations[layers_left]
            delta_j = 0
            for neuron, i in activation:
                delta_j = delta_j + delta(layers_left, i, item, output)*weights[layers_left][i]*neuron(1-neuron)
        return delta_j

    @staticmethod
    def loss():
        pass

nn = NeuralNetwork([2, 2, 1])

nn.feedforward()