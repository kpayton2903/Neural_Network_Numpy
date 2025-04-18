import numpy as np

class Layers:
    '''
    A class used to create the layers within our neural network. Each instance of this class is an individual layer,
    starting with the first hidden layer and ending with the output layer.

    Attributes:
    -----------
    weights : np.ndarray
        An n-dimensional array containing the weights that are multiplied by the input values in the forward propagation step.
        These weights are optimized during training through back propagation.
    bias : np.ndarray
        An n-dimensional array containing the biases that are added at each neuron. These are also optimized during training
        through back propagation.
    activ_func : object
        The activation function that will be used to provide non-linearity to our network. The input should be one of the
        classes outlined in the Activations.py file.
    is_output : boolean
        A boolean identifying if the specified layer is the output layer. This is used so we can utilize a simplification
        from the chain rule between cross-entropy and the Softmax activation function.
    input : np.ndarry
        An n-dimensional array that is passed as the input to the specified layer.
    Z : np.ndarray
        The input array after it has been transformed by the weights and biases.
    A : np.ndarray
        The Z array after it has been transformed by the activation function contained in activ_func.
    '''

    def __init__(self, input_size:int, output_size:int, activation_function=None):
        '''
        Initializes a Layer with random weights and biases, a given activation function, and set is_output to false by default.
        The initialized weights has shape [output_size, input_size] and utilize a He initialization so that the values have
        a normal distribution with mean 0 and variance 2 / input_size. The initialized biases has shape [output_size, 1] and
        have values with a standard normal distribution.

        Parameters:
        -----------
        input_size : int
            An integer that describes the amount of input variables that will be coming in to the specified layer.
        output_size : int
            An integer that describes the amount of output variables that will be returned from the specified layer.
        activation_function : object
            An activation function class that will initialize our activ_func attribute.
        '''

        self.weights = np.random.randn(output_size, input_size) * np.sqrt(2.0 / input_size)
        self.bias = np.random.randn(output_size, 1)
        self.activ_func = activation_function
        self.is_output = False

    def forward(self, input:np.ndarray) -> np.ndarray:
        '''
        Given some input, runs the forward propagation step by doing the dot product of our weights and input,
        then adding the bias. These new values are then sent to the activation function.

        Parameters:
        -----------
        input : np.ndarray
            An n-dimensional array that is passed as the input to the specified layer and initializes the input
            attribute.
        
        Returns:
        --------
        np.ndarry
            An n-dimensional array that has been transformed by the weights and biases, and potential activation
            function.
        '''

        self.input = input
        Z = np.dot(self.weights, self.input) + self.bias
        self.Z = Z

        if (self.activ_func != None):
            self.A = self.activ_func(Z)
            return self.A
        else:
            return Z
        
    def backward(self, output_grad:np.ndarray, learning_rate:float) -> np.ndarray:
        '''
        Performs the backpropagation algorithm by running gradient descent on the weights and biases, using
        the derivative of the loss with respect to said weights and biases as the gradient.

        Parameters:
        -----------
        output_grad : np.ndarray
            The gradient we can reuse from the previous layers. In the case of the output layer, this is
            simply the derivative of the loss.
        learning_rate : float
            The learning rate used in gradient descent.

        Returns:
        --------
        np.ndarray
            An n-dimensional array of gradients we can reuse for the next layer in backpropagation.
        '''

        observations = self.input.shape[1]

        # Currently this is designed in a way to use softmax only as the output activation function
        # We are utilizing a special property of the gradient of the loss with respect to the
        # input to this layer.
        if (self.activ_func is not None and not self.is_output):
            delta = output_grad * self.activ_func.gradient(self.Z)
        else:
            delta = output_grad

        # Find the gradient of the loss with respect to the weights and average it increase stability
        dw = np.dot(delta, self.input.T) / observations
        # Find the gradient of the loss with respect to the bias and average it as well
        db = np.sum(delta, axis=1, keepdims=True) / observations

        # Perform this dot product for the next layer's backpropagation
        new_output_grad = np.dot(self.weights.T, delta)

        # Adjust weights and biases
        self.weights -= learning_rate * dw
        self.bias -= learning_rate * db
        
        return new_output_grad