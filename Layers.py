import numpy as np
from typing import Callable, Optional

class Layers:
    def __init__(self, input_size:int, output_size:int, activation_function=None):
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)
        self.activ_func = activation_function

    def forward(self, input):
        self.input = input
        Z = np.dot(self.weights, self.input) + self.bias
        self.Z = Z

        if (self.activ_func != None):
            self.A = self.activ_func(Z)
            return self.A
        else:
            return Z
        
    def backward(self, output_grad, learning_rate:float):
