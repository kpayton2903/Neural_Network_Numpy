import numpy as np

class ReLu:
    def __call__(self, Z):
        '''
        The Re-Lu function returns only the positive values from its input and replaces the negatives ones
        with zero.

        Z(array): The transformed input values at our current hidden layer
        '''
        return np.maximum(0, Z)
    
    def gradient(self, Z):
        Y  = self(Z)
        Y[Y >= 0] = 1
        return Y

class Softmax:
    def __call__(self, Z):
        '''
        The Softmax function only returns values between zero and one.

        Z(array): The transformed input values at our current hidden layer
        '''
        return np.exp(Z) / np.sum(np.exp(Z))
    
    def gradient(self, Z):
        Y = self(Z).reshape(-1, 1)
        return np.diagflat(Y) - np.dot(Y, Y.T)

class Sigmoid:
    def __call__(Z):
        '''
        The Sigmoid function
        '''
        return 1 / (1 + np.exp(-Z))
    
    def gradient(self, Z):
        Y = self(Z)
        return Y * (1 - Y)
