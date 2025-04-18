import numpy as np

class ReLu:
    def __call__(self, Z):
        '''
        The Re-Lu function returns only the positive values from its input and replaces the negatives ones
        with zero.

        Parameters:
        -----------
        Z : np.ndarray
            An n-dimensional array to be transformed. In the case of our neural network, this is the array that
            has been transformed by the weights and biases between the layers.

        Returns:
        --------
        np.ndarray
            A scaled n-dimensional array with only positive values.
        '''

        return np.maximum(0, Z)
    
    def gradient(self, Z):
        '''
        This is simply just the gradient, or derivative, of the Re-Lu function described above.
        '''
        
        Y  = self(Z)
        Y[Y >= 0] = 1
        return Y

class Softmax:
    def __call__(self, Z):
        '''
        The Softmax function returns only values between zero and one, with all the elements in the returned
        array adding up to 1. This is often used for multi-class prediction problems.

        Parameters:
        -----------
        Z : np.ndarray
            An n-dimensional array to be transformed. In the case of our neural network, this is the array that
            has been transformed by the weights and biases between the layers.

        Returns:
        --------
        np.ndarray
            A scaled n-dimensional array with only values between zero and one. These values are viwed as the
            probability distribution of all the classes. We often choose the largest probability class to
            be the predicted value.
        '''

        # Rather than just simply calculating exp(Z), we subtract the maximum value to ensure numerical stability
        expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
        return expZ / np.sum(expZ, axis=0, keepdims=True)

    def gradient(self, Z):
        '''
        This is simply just the gradient, or derivative, of the Softmax function described above.
        '''

        Y = self(Z).reshape(-1, 1)
        return np.diagflat(Y) - np.dot(Y, Y.T)

class Sigmoid:
    def __call__(self, Z):
        '''
        The Sigmoid function returns only values between zero and one, with all the elements in the returned
        array adding up to 1. This is often used for multi-class prediction problems.

        Parameters:
        -----------
        Z : np.ndarray
            An n-dimensional array to be transformed. In the case of our neural network, this is the array that
            has been transformed by the weights and biases between the layers.

        Returns:
        --------
        np.ndarray
            A scaled n-dimensional array with only values between zero and one. These values are viwed as the
            probability distribution of all the classes. We often choose the largest probability class to
            be the predicted value.
        '''

        return 1 / (1 + np.exp(-Z))
    
    def gradient(self, Z):
        '''
        This is simply just the gradient, or derivative, of the Sigmoid function described above.
        '''

        Y = self(Z)
        return Y * (1 - Y)
