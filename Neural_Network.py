import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Neural_Network:
    def __init__(self, layer_sizes:tuple, learning_rate:float=0.01, hidden_activ:str='re-lu', out_activ:str='softmax'):
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.hidden_activ = hidden_activ
        self.output_activ = out_activ

    def init_weights_biases(self):
        '''
        Initializes the weights and biases for our network with random values between -0.5 and 0.5

        inputs(integer): The number of input values we will have in our network
        '''

        weights = [np.random.randn(self.layer_sizes[1], self.layer_sizes[0])]
        biases = [np.random.randn(self.layer_sizes[1], 1)]

        # If we have a hidden layer
        if len(self.layer_sizes) > 2:
            for i in range(len(self.layer_sizes) - 2):
                weights.append(np.random.randn(self.layer_sizes[i+1], self.layer_sizes[i]))
                biases.append(np.random.randn(self.layer_sizes[i+1], 1))
        
        self.weights = weights
        self.biases = biases

    def relu(Z):
        '''
        The Re-Lu function returns only the positive values from its input and replaces the negatives ones
        with zero.

        Z(array): The transformed input values at our current hidden layer
        '''
        return np.maximum(0, Z)
    
    def softmax(Z):
        '''
        The Softmax function only returns values between zero and one.

        Z(array): The transformed input values at our current hidden layer
        '''
        return np.exp(Z) / np.sum(np.exp(Z))
    
    def sigmoid(Z):
        '''
        The Sigmoid function
        '''
        return 1 / (1 + np.exp(-Z))

    def forward(self, X):
        '''
        Performs the forward computation through our network, multiplying the inputs by the weights and adding
        the biases at each hidden layer. Also runs the specified activation functions at each hidden layer.

        '''
        Z = [np.dot(self.weights[0], X) + self.biases[0]]
        A = []

        if (len(self.layer_sizes) > 2):
            A.append(self.relu(Z))

            for i in range(len(self.layer_sizes) - 2):
                Z.append(np.dot(self.weights[i + 1], Z[i]) + self.biases[i + 1])
                A.append(self.relu(Z[i + 1]))

            Z.append(np.dot(self.weights[-1], Z[-1]))
        
        if (self.output_activ == 'softmax'):
            A.append(self.softmax(Z[-1]))
        elif (self.output_activ == 'sigmoid'):
            A.append(self.sigmoid(Z[-1]))

'''
    def back_prop(self):


    def train(self, X, Y):
'''


if __name__ == "__main__":
    model = Neural_Network(layer_sizes=(10))
    model.init_weights_biases(20)
    print(model.weights)
    print(model.biases)