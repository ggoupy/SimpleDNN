#!/usr/bin/env python
# coding: utf-8

# Package imports
import time
import numpy as np
import h5py
from PIL import Image



class DNN:
    
    def __init__(self, layer_dims, parameters = None):
        """
        Initialize the deep neural network.
        
        Arguments:
        layer_dims -- python array (list) containing the dimensions of each layer in the network [input_layer, hidden_layers, output_layer]
        parameters -- optional, weights and bias for each layers (if not given, init to random)
        """
        np.random.seed(1)      
        self.layer_dims = layer_dims 
        L = len(layer_dims) # Number of layers in the network
        self.L = L - 1 # input layer is not counted
        self.parameters = {}
        if parameters == None: # Random weights init
            for l in range(1, L): #We started from 1 as first element in list is input layer
                self.parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) / np.sqrt(layer_dims[l-1]) #*0.01 //other way to do so
                self.parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
                assert(self.parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
                assert(self.parameters['b' + str(l)].shape == (layer_dims[l], 1))
        else:
            self.parameters = parameters


    def __sigmoid(self, Z):
        """
        Implement the sigmoid activation function for the deep neural network.
        """
        return 1/(1+np.exp(-Z))


    def __relu(self, Z):
        """
        Implement the RELU activation function for the deep neural network.
        """
        A = np.maximum(0,Z)
        assert(A.shape == Z.shape)
        return A


    def __sigmoid_backward(self, dA, cache):
        """
        Implement the backward propagation for a single SIGMOID unit.

        Arguments:
        dA -- post-activation gradient, of any shape
        cache -- 'Z' where we store for computing backward propagation efficiently

        Returns:
        dZ -- Gradient of the cost with respect to Z
        """    
        Z = cache 
        s = 1/(1+np.exp(-Z))
        dZ = dA * s * (1-s)
        assert (dZ.shape == Z.shape)
        return dZ
        
        
    def __relu_backward(self, dA, cache):
        """
        Implement the backward propagation for a single RELU unit.

        Arguments:
        dA -- post-activation gradient, of any shape
        cache -- 'Z' where we store for computing backward propagation efficiently

        Returns:
        dZ -- Gradient of the cost with respect to Z
        """
        Z = cache
        dZ = np.array(dA, copy=True) # just converting dz to a correct object.
        # When z <= 0, you should set dz to 0 as well
        dZ[Z <= 0] = 0   
        assert (dZ.shape == Z.shape)
        return dZ

    
    def __linear_activation_forward(self, A_prev, W, b, activation):
        """
        Implement the forward propagation for the LINEAR->ACTIVATION layer.

        Arguments:
        A_prev -- activations from previous layer (or input data) of shape (size of previous layer, number of examples)
        W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
        b -- bias vector, numpy array of shape (size of the current layer, 1)
        activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

        Returns:
        A -- the output of the activation function, also called the post-activation value 
        cache -- a python dictionary containing "linear_cache" and "activation_cache";
                 stored for computing the backward pass efficiently
        """
        # Linear computation 
        Z = np.dot(W, A_prev) + b
        linear_cache = (A_prev, W, b)
        
        # Activation computation
        if activation == "sigmoid":
            A = self.__sigmoid(Z)
        elif activation == "relu":
            A = self.__relu(Z)
        assert (A.shape == (W.shape[0], A_prev.shape[1]))
        activation_cache = Z

        cache = (linear_cache, activation_cache)

        return A, cache
        
        
    def __linear_activation_backward(self, dA, cache, activation):
        """
        Implement the backward propagation for the LINEAR->ACTIVATION layer.
        
        Arguments:
        dA -- post-activation gradient for the layer
        cache -- tuple of values (linear_cache, activation_cache) for computing backward propagation efficiently
        activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
        
        Returns:
        dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dW -- Gradient of the cost with respect to W (current layer l), same shape as W
        db -- Gradient of the cost with respect to b (current layer l), same shape as b
        """
        linear_cache, activation_cache = cache
        
        dZ = self.__relu_backward(dA, activation_cache) if activation == "relu" else self.__sigmoid_backward(dA, activation_cache)
        
        A_prev, W, b = linear_cache
        m = A_prev.shape[1]
        dW = 1./m * np.dot(dZ,A_prev.T)
        db = 1./m * np.sum(dZ, axis = 1, keepdims = True)
        dA_prev = np.dot(W.T,dZ)
        assert (dA_prev.shape == A_prev.shape)
        assert (dW.shape == W.shape)
        assert (db.shape == b.shape)
      
        return dA_prev, dW, db

    
    def __forward_propagation(self, X):
        """
        Implement the forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation.
        
        Arguments:
        X -- data, numpy array of shape (number of features, number of examples)
        
        Returns:
        AL -- last post-activation value
        caches -- list of caches containing:
                    every cache of linear_relu_forward() (there are L-1 of them, indexed from 0 to L-2)
                    the cache of linear_sigmoid_forward() (there is one, indexed L-1)
        """
        caches = []
        A = X
        
        # Hidden laters, [LINEAR -> RELU]*(L-1)
        for l in range(1, self.L):
            A_prev = A 
            A, cache = self.__linear_activation_forward(A_prev, self.parameters['W' + str(l)], self.parameters['b' + str(l)], activation = "relu")
            caches.append(cache)
        
        # output layer, LINEAR -> SIGMOID
        AL, cache = self.__linear_activation_forward(A, self.parameters['W' + str(self.L)], self.parameters['b' + str(self.L)], activation = "sigmoid")
        caches.append(cache)
        
        assert(AL.shape == (1,X.shape[1]))
                
        return AL, caches
        
        
    def __backward_propagation(self, AL, Y, caches):
        """
        Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID computation.
        
        Arguments:
        AL -- probability vector, output of the forward propagation
        Y -- label vector
        caches -- list of caches from forward propagation
        
        Returns:
        grads -- A dictionary with the gradients
        """
        grads = {}
        Y = Y.reshape(AL.shape) # Y is now the same shape as AL
        
        # Initializing the backward propagation
        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
        
        # Output layer, (SIGMOID -> LINEAR) gradients
        current_cache = caches[self.L-1]
        grads["dA" + str(self.L-1)], grads["dW" + str(self.L)], grads["db" + str(self.L)] = self.__linear_activation_backward(dAL, current_cache, activation = "sigmoid")
        
        # Hidden layers, (RELU -> LINEAR) gradients
        for l in reversed(range(self.L-1)): # L-2 to 0
            
            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = self.__linear_activation_backward(grads["dA" + str(l + 1)], current_cache, activation = "relu")
            grads["dA" + str(l)] = dA_prev_temp
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp

        return grads
        
        
    def __compute_cost(self, AL, Y):
        """
        Implement the cost function for the deep neural network.

        Arguments:
        AL -- probability vector corresponding to label predictions, shape (1, number of examples)
        Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

        Returns:
        cost -- cross-entropy cost
        """
        m = Y.shape[1] # Number of examples

        # Compute loss from aL and y
        cost = (1./m) * (-np.dot(Y,np.log(AL).T) - np.dot(1-Y, np.log(1-AL).T))

        # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17)
        cost = np.squeeze(cost)
        assert(cost.shape == ())
   
        return cost


    def __update_parameters(self, grads, learning_rate):
        """
        Update parameters of the deep neural network using gradient descent.
        
        Arguments:
        grads -- python dictionary containing the gradients, output of backward propagation
        learning_rate -- learning rate of the gradient descent update rule
        """
        # Update rule for each parameter
        for l in range(1, self.L+1):
            self.parameters["W" + str(l)] = self.parameters["W" + str(l)] - learning_rate * grads["dW" + str(l)]
            self.parameters["b" + str(l)] = self.parameters["b" + str(l)] - learning_rate * grads["db" + str(l)]
    
    
    def train(self, X, Y, learning_rate = 0.0075, num_iterations = 3000, print_cost = False):
        """
        Train the deep neural network with given dataset X associated to labels Y.
        
        Arguments:
        X -- dataset, numpy array of shape (number of features, number of examples)
        Y -- label vector of shape (1, number of examples)
        learning_rate -- learning rate of the gradient descent update rule
        num_iterations -- number of iterations of the optimization loop
        print_cost -- if True, it prints the cost every 100 steps
        """
        np.random.seed(1)
        
        # Loop (gradient descent)
        for i in range(0, num_iterations):

            # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID
            AL, caches = self.__forward_propagation(X)
            
            # Compute cost
            cost = self.__compute_cost(AL, Y)
        
            # Backward propagation
            grads = self.__backward_propagation(AL, Y, caches)
     
            # Update parameters
            self.__update_parameters(grads, learning_rate)
                    
            # Print the cost every 100 iterations
            if print_cost and i % 100 == 0 or i == num_iterations - 1:
                print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))


    def predict(self, X, y = []):
        """
        Predict the output for given input of the deep neural network.
        
        Arguments:
        X -- input dataset of examples you would like to label
        Y -- vector labels of input, optional
        
        Returns:
        p -- predictions for the given dataset X
        """
        m = X.shape[1]
        p = np.zeros((1,m))
        
        # Forward propagation
        probas, _ = self.__forward_propagation(X)

        # Convert probas to 0/1 predictions
        for i in range(0, probas.shape[1]):
            p[0,i] = 1 if probas[0,i] > 0.5 else 0
        
        #print results
        #print ("predictions: " + str(p))
        #print ("true labels: " + str(y))
        if y != []:
            print("Accuracy: "  + str(np.sum((p == y)/m)))
            
        return p
        
        
    def save_to_file(self, file = 'dnn_model.npy'):
        """
        Save the DNN model in a numpy file.
        
        Arguments:
        file -- path of the file, must use .npy format
        """
        np.savez(file, name1=self.layer_dims, name2=self.parameters)
        
        
    def load_from_file(self, file = 'dnn_model.npy'):
        """
        Load a DNN model from a numpy file.
        
        Arguments:
        file -- path of the file, must use .npy format
        """  
        data = np.load(file, allow_pickle='TRUE').item()
        self.layer_dims = data['name1']
        self.parameters = data['name2']
        self.L = len(self.layer_dims) - 1
        
        
    @staticmethod
    def create_from_file(file):
        """
        Create a DNN model from a numpy file.
        
        Arguments:
        file -- path of the file, must use .npy format
        """
        data = np.load(file, allow_pickle='TRUE')
        layer_dims = data['name1']
        parameters = data['name2']
        return dnn(layer_dims, parameters)
