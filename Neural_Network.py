import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Neural_Network:
    def __init__(self, layer_sizes:tuple, learning_rate:float=0.01, hidden_activ:str='re-lu', out_activ:str='softmax'):
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.hidden_activ = hidden_activ
        self.output_activ = out_activ

'''
    def train(self, X, Y):
'''


if __name__ == "__main__":
    model = Neural_Network(layer_sizes=(10))
    model.init_weights_biases(20)
    print(model.weights)
    print(model.biases)