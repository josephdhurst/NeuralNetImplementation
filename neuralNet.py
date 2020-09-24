
import numpy as np

class Loss(object):
    
    def __call__(self, predicted, actual):
        """Calculates the loss as a function of the prediction and the actual.
        
        Args:
          predicted (np.ndarray, float): the predicted output labels
          actual (np.ndarray, float): the actual output labels
          
        Returns: (float) 
          The value of the loss for this batch of observations.
        """
        raise NotImplementedError
        
    def derivative(self, predicted, actual):
        """The derivative of the loss with respect to the prediction.
        
        Args:
          predicted (np.ndarray, float): the predicted output labels
          actual (np.ndarray, float): the actual output labels
          
        Returns: (np.ndarray, float) 
          The derivatives of the loss.
        """
        raise NotImplementedError

class SquaredErrorLoss(Loss):
    def __call__(self, predicted, actual):
        return np.mean((predicted - actual) ** 2)
    
    def derivative(self, predicted, actual):
        return 2 * np.sum(predicted - actual)

class ActivationFunction(object):
        
    def __call__(self, a):
        """Applies activation function to the values in a layer.
        
        Args:
          a (np.ndarray, float): the values from the previous layer (after 
            multiplying by the weights.
          
        Returns: (np.ndarray, float) 
          The values h = g(a).
        """
        return a
    
    def derivative(self, h):
        """The derivatives as a function of the outputs at the nodes.
        
        Args:
          h (np.ndarray, float): the outputs h = g(a) at the nodes.
          
        Returns: (np.ndarray, float) 
          The derivatives dh/da.
        """
        return 1

class ReLU(ActivationFunction):
    def __call__(self, a):
        return (a > 0) * a
    
    def derivative(self, h):
        return (h > 0) * 1


class Sigmoid(ActivationFunction):
    def __call__(self, a):
        return 1 / (1 + np.exp(a))
    
    def derivative(self, h):
        return h * (1 - h)

def matrixMean(matrixList):
    matrixSum = matrixList[0]
    for matrix in matrixList[1:]:
        matrixSum += matrix
    return matrixSum / len(matrixList)

class Layer(object):
    """A data structure for a layer in a neural network.
    
    Attributes:
      num_nodes (int): number of nodes in the layer
      activation_function (ActivationFunction)
      values_pre_activation (np.ndarray, float): most recent values
        in layer, before applying activation function
      values_post_activation (np.ndarray, float): most recent values
        in layer, after applying activation function
    """
    
    def __init__(self, num_nodes, activation_function=ActivationFunction()):
        self.num_nodes = num_nodes
        self.activation_function = activation_function
        
    def get_layer_values(self, values_pre_activation):
        """Applies activation function to values from previous layer.
        
        Stores the values (both before and after applying activation 
        function)
        
        Args:
          values_pre_activation (np.ndarray, float): 
            A (batch size) x self.num_nodes array of the values
            in layer before applying the activation function
        
        Returns: (np.ndarray, float)
            A (batch size) x self.num_nodes array of the values
            in layer after applying the activation function
        """
        self.values_pre_activation = values_pre_activation
        self.values_post_activation = self.activation_function(
            values_pre_activation
        )
        return self.values_post_activation

        
class FullyConnectedNeuralNetwork(object):
    """A data structure for a fully-connected neural network.
    
    Attributes:
      layers (Layer): A list of Layer objects.
      loss (Loss): The loss function to use in training.
      learning_rate (float): The learning rate to use in backpropagation.
      weights (list, np.ndarray): A list of weight matrices,
        length should be len(self.layers) - 1
      biases (list, float): A list of bias terms,
        length should be equal to len(self.layers)
    """
    
    def __init__(self, layers, loss, learning_rate):
        self.layers = layers
        self.loss = loss
        self.learning_rate = learning_rate
        
        # initialize weight matrices and biases to zeros
        self.weights = []
        self.biases = []
        for i in range(1, len(self.layers)):
            self.weights.append(
                np.random.normal(2, 0.5, size = (self.layers[i - 1].num_nodes, self.layers[i].num_nodes))
            )
            self.biases.append(np.zeros(self.layers[i].num_nodes))
    
    def feedforward(self, inputs):
        """Predicts the output(s) for a given set of input(s).
        
        Args:
          inputs (np.ndarray, float): A (batch size) x self.layers[0].num_nodes array
          
        Returns: (np.ndarray, float) 
          An array of the predicted output labels, length is the batch size
        """
        # TODO: Implement feedforward prediction.
        # Make sure you use Layer.get_layer_values() at each layer to store the values
        # for later use in backpropagation.
        
        currentOutput = self.layers[0].get_layer_values(inputs)
        # iterate over the hidden layers
        
        for ix, weights in enumerate(self.weights):
            layerInput =  np.matmul(currentOutput, weights) + self.biases[ix]
            currentOutput = self.layers[ix + 1].get_layer_values(layerInput)
        
        return currentOutput
        
        
    def backprop(self, predicted, actual):
        """Updates self.weights and self.biases based on predicted and actual values.
        
        This will require using the values at each layer that were stored at the
        feedforward step.
        
        Args:
          predicted (np.ndarray, float): An array of the predicted output labels
          actual (np.ndarray, float): An array of the actual output labels
        """
        # Calulate the initial error for each data point
        self.layers[-1].error = []
        for ix in range(len(predicted)):
            self.layers[-1].error.append(np.array([[self.loss.derivative(predicted[ix], 
                                                                         actual[ix])]]))
        # start with the last layer before the output layer
        # BackPropogate the error
        index = -2
        while index > -1 * len(self.layers):
            lastLayer = self.layers[index]
            # calculate the error for each data point
            lastLayer.error = []
            for ix in range(len(predicted)):
                activationDerivative = np.array([lastLayer.activation_function.derivative(
                    lastLayer.values_post_activation[ix])])
                lastLayer.error.append((np.matmul(self.weights[index + 1], 
                                             self.layers[index + 1].error[ix]
                                            ) * activationDerivative.T).T)
            index -= 1
        
        index = -1
        while index > -1 * len(self.layers):
            gradients = []
            for ix in range(len(predicted)):
                gradients.append(np.matmul(self.layers[index].error[ix].T,
                                     np.array([self.layers[index - 1].values_post_activation[ix]])))
            
            gradient = matrixMean(gradients)
            self.weights[index] = self.weights[index] - self.learning_rate * gradient.T
            meanError = matrixMean(self.layers[index].error)
            self.biases[index] = self.biases[index] - meanError * self.learning_rate
            index -= 1
        
    def train(self, inputs, labels):
        """Trains neural network based on a batch of training data.
        
        Args:
          inputs (np.ndarray): A (batch size) x self.layers[0].num_nodes array
          labels (np.ndarray): An array of ground-truth output labels, 
            length is the batch size.
        """
        predicted = self.feedforward(inputs)
        self.backprop(predicted, labels)
        return self.loss(predicted, labels)

