import numpy as np

class NeuralNetwork:
    def __init__(self, shape):
        self.shape = shape
        self.weights, self.biases = self.xavier(shape)
        self.input = np.random.normal(loc = 0.0, scale=1.0, size = (shape[0], 1))

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
        for i in range(len(self.shape) - 1):
            bias = self.biases[i]
            weight_matrix = self.weights[i]
            next_activation = self.ReLU(weight_matrix*activation + bias)
            activation = next_activation    
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
            new_weights, new_biases = self.backpropagate(output)
    
    @staticmethod
    def backpropagate(output):
        
        return (new_weights, new_biases)

    @staticmethod
    def loss():
        pass

nn = NeuralNetwork([2, 2, 1])

print(nn.feedforward())