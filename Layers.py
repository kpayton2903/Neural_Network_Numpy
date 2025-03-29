import numpy as np
from typing import Callable, Optional

class Layers:
    def __init__(self, input_size:int, output_size:int, activation_function:Optional[Callable]=None):
        self.weights = np.randn(output_size, input_size)
        self.bias = np.randn(output_size, 1)
        self.activ_func = activation_function

    def forward(self, input):
        self.input = input
        output = np.dot(self.weights, self.input) + self.bias
        if (self.activ_func != None):
            return self.activ_func(output)
        else:
            return output
