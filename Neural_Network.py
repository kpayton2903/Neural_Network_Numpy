import numpy as np
from Layers import Layers
from Activations import ReLu, Softmax, Sigmoid

class Neural_Network:
    '''
    A class used to create a neural network which in turn can be used to perform predictions after it has been trained
    on a given dataset.

    Attributes:
    layer_sizes : tuple
        The size of each layer in the network, with the first being the size of the input layer, and the last being
        the size of the output layer.
    learning_rate : float
        The learning rate used during the backpropagation step.
    '''
    def __init__(self, layer_sizes:tuple,
                 learning_rate:float=0.0001,
                 hidden_activ:str='relu',
                 out_activ:str='softmax',
                 seed:int=100):
        '''
        Initializes the neural network with a tuple containing the layer sizes, a learning rate, and creates an
        empty list that will contain the layers once their classes are instantiated. Also sets the seed for the
        random processes that occur in the layers so we can reproduce out results.

        Parameters:
        -----------
        layer_sizes : tuple
            A tuple containing the sizes for each layer in the network, including the input and output.
        learning_rate : float
            A floating point number that will be used as the learning rate during gradient descent in the
            backpropagation step.
        layers : list
            A list containing each of the layers in the network.
        '''
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.layers = []
        np.random.seed(seed)

        for i in range(1, len(layer_sizes)):
            # Initializes the activation function classes for our layers
            if i == (len(layer_sizes) - 1):
                if out_activ.lower() == 'softmax':
                    activ_func = Softmax()
                elif out_activ.lower() == 'sigmoid':
                    activ_func = Sigmoid()
            else:
                if hidden_activ.lower() == 'relu':
                    activ_func = ReLu()
                elif hidden_activ.lower() == 'sigmoid':
                    activ_func = Sigmoid()

            # Initialize and append a layer to our network
            self.layers.append(Layers(layer_sizes[i-1], layer_sizes[i], activ_func))
            if i == (len(layer_sizes) - 1):
                # Specifies if the given layer is the output layer
                self.layers[-1].is_output = True

    def forward(self, X:np.ndarray) -> np.ndarray:
        '''
        Iterates through the layers in our network and performs the forward propagation step to make a prediction based
        on our input.

        Parameters:
        -----------
        X : np.ndarray
            An input array to be sent through the network and transformed to make a prediction.

        Returns:
        --------
        np.ndarray
            An array of predicted values.
        '''

        output = X
        for layer in self.layers:
            output = layer.forward(output)
        
        return output
    
    def cross_entropy_loss(self, Y_pred:np.npdarray, Y_true:np.ndarray) -> float:
        '''
        Computes the cross-entropy loss to see how well the model is performing.

        Parameters:
        -----------
        Y_pred : np.ndarray
            An array of predicted values found from the forward propagation step.
        Y_true : np.ndarray
            An array of the true values for Y.

        Returns:
        --------
        float
            A floating point value that is a measure of the model's performance.
        '''

        # Clips values on the edges of our array to avoid issues with Y_pred being 0.
        Y_pred = np.clip(Y_pred, 1e-12, 1.0 - 1e-12)
        return -np.sum(Y_true * np.log(Y_pred))
    
    def predict(self, X:np.ndarray) -> int:
        '''
        Makes a prediction based on the values found from the forward propagation step.

        Parameters:
        -----------
        X : np.ndarray
            An input array to be sent through the network and transformed to make a prediction.
        
        Returns:
        --------
        int
            The index of the predicted class with the greatest value, or the class with the largest
            percentage.
        '''
        Y_pred = self.forward(X)
        return np.argmax(Y_pred,axis=0)
    
    def accuracy(self, Y_pred:np.ndarray, Y_true:np.ndarray) -> float:
        '''
        Calculates the accuracy of our predictions by comparing predicted values and the true values.

        Parameters:
        -----------
        Y_pred : np.ndarray
            An array of predicted values found from the forward propagation step.
        Y_true : np.ndarray
            An array of the true values for Y.

        Returns:
        --------
        float
            A floating point value representing the accuracy of the predictions.
        '''
        return np.mean(Y_pred == Y_true)
    
    def train(self, X:np.ndarray, Y:np.ndarray, epochs:int = 1000):
        '''
        Trains the model by performing forward and backward propagation for a set number of epochs. This
        slowly makes the weights and biases better suited for our dataset.

        Parameters:
        -----------
        X : np.ndarray
            An input array to be sent through the network and used to train the model
        Y : np.ndarray
            The desired output values to get from inputting X into the model.
        epochs : int
            The number of iterations to be performed during training.

        Returns:
        --------
        list, list
            Two lists, with the first containing the losses, and the second containing the accuracies at each epoch
        '''
        losses=[]
        accuracies = []
        n_samples = X.shape[1]

        for epoch in range(epochs + 1):
            # Make a prediction using forward propagation
            Y_pred = self.forward(X)

            # Calculate the total loss and divide it by the samples to get the average
            loss = self.cross_entropy_loss(Y_pred, Y) / n_samples
            losses.append(loss)

            # Calculate the accuracy for the given epoch
            accuracy = np.mean(np.argmax(Y_pred, axis=0) == np.argmax(Y, axis=0))
            accuracies.append(accuracy)

            # At every 100 epochs, print the loss and accuracy
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
                print(f"Accuracy: {accuracy:.4f}")

            # The initial gradient input to the backward method calculated as described in the Layers class
            init_grad = Y_pred - Y

            # Perform backpropagation at each layer, altering the weights and biases
            for layer in reversed(self.layers):
                init_grad = layer.backward(init_grad, self.learning_rate)
            
        return losses, accuracies
    
    def train_batches(self, X:np.ndarray, Y:np.ndarray, epochs:int = 50, batch_size:int = 50):
        '''
        Trains the model by performing forward and backward propagation for a set number of epochs. This
        slowly makes the weights and biases better suited for our dataset. Utilizes batches by splitting 
        up the X and Y arrays to optimize performance with large datasets.

        Parameters:
        -----------
        X : np.ndarray
            An input array to be sent through the network and used to train the model
        Y : np.ndarray
            The desired output values to get from inputting X into the model.
        epochs : int
            The number of iterations to be performed during training.
        batch_size : int
            The size of each batch we use 

        Returns:
        --------
        list, list
            Two lists, with the first containing the losses, and the second containing the accuracies at each epoch
        '''
        losses=[]
        accuracies = []
        n_samples = X.shape[1]

        for epoch in range(epochs + 1):
            epoch_loss = 0
            amount_correct = 0

            for i in range(0, n_samples, batch_size):
                # Split up our X and Y based on the batch size provided
                X_batch = X[:, i:i+batch_size]
                Y_batch = Y[:, i:i+batch_size]

                # Make a prediction using forward propagation
                Y_pred = self.forward(X_batch)

                # Calculate the total loss and amount correct for the batch
                epoch_loss += self.cross_entropy_loss(Y_pred, Y_batch)
                amount_correct += np.sum(np.argmax(Y_pred, axis=0) == np.argmax(Y_batch, axis=0))

                # The initial gradient input to the backward method calculated as described in the Layers class
                init_grad = Y_pred - Y_batch

                # Perform backpropagation at each layer, altering the weights and biases
                for layer in reversed(self.layers):
                    init_grad = layer.backward(init_grad, self.learning_rate)
        
            loss = epoch_loss / n_samples
            losses.append(loss)
            accuracy = amount_correct / n_samples
            accuracies.append(accuracy)

            # At every epoch, print the loss and accuracy
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
            print(f"Accuracy: {accuracy:.4f}")

        return losses, accuracies