from utils import init_parameters, ff_predict
from typing import List, Callable
import numpy as np


class model:
    def __init__(self, hidden_layer_dimensions: List[int], input_data: np.ndarray, target_labels: np.ndarray,
                 activation_functions: List[Callable],
                 training_iterations: int, learning_rate: float, regularization_coefficient: float):
        """
        Initialize the Neural Network.
        :param hidden_dim: A list of integers, where each integer represents the number of neurons in a layer.
        :param X: The input data.
        :param y: The target labels.
        :param activations: A list of tuples, where each tuple contains the activation function, its derivative and
        its name.
        :param epochs: The number of training iterations.
        :param alpha: The learning rate.
        :param Lambda: The regularization coefficient.
        """
        output_dimensions = np.unique(target_labels).size
        self.layer_dimensions = [input_data.shape[1]] + hidden_layer_dimensions + [output_dimensions]
        self.weights = [init_parameters(input_dimension, output_dimension) for input_dimension, output_dimension in zip(self.layer_dimensions, self.layer_dimensions[1:])]
        self.input_data = input_data
        self.target_labels = target_labels
        self.activation_functions = activation_functions
        self.training_iterations = training_iterations
        self.learning_rate = learning_rate
        self.regularization_coefficient = regularization_coefficient
        self.number_of_samples = input_data.shape[0]



    def feed_forward():
        a_arr = []
        a0 = self.input_data.reshape(-1, 1)
        i = 0
        for Theta in self.weights:
            ones = np.ones((1, 1))
            a0 = np.concatenate((ones, a0), axis=0)
            z = np.dot(Theta, a0)

            a0 = self.activation_functions[i](z)
            a_arr.append(a0)
            i += 1
        a_arr.insert(0, X)
        a_arr[:-1] = [np.insert(a_, 0, 1) for a_ in a_arr[:-1]]  ## add 1 bias
        a_arr = [a_.reshape(a_.shape[0], 1) for a_ in a_arr]
        return a_arr


    def train(self):
        """
        Train the Neural Network using backpropagation.
        :return: The cost and the weights of the Neural Network.

        """
        predicted_labels = np.zeros((self.number_of_samples, 1))
        cost = 0
        epsilon = 1e-8

        for iteration  in range(self.training_iterations):
            weight_gradients = [np.zeros(weight.shape) for weight in self.weights]
            random_indices = np.random.permutation(self.number_of_samples)
            number_of_layers = len(self.activation_functions)

            for index in random_indices:

                J = 0
                activations = self.feed_forward()
                ### start Backward propagation

                # Assigning 1 to the binary digit according to the class (label) of the input
                binary_target = np.zeros(activations[-1].shape)
                binary_target[self.target_labels[index]] = 1

                cost += (-1) * (np.dot(binary_target.T, np.log(np.maximum(activations[-1], epsilon))) +
                             np.dot((1 - binary_target).T, np.log(np.maximum(1 - activations[-1], epsilon))))
                # output layer
                current_delta = (activations[-1] - binary_target)
                weight_gradients[-1] += np.dot(current_delta, activations[-2].T)

                for layer in range(number_of_layers - 1, 0, -1):
                    activation_derivative = self.activation_functions[layer](activations[layer])
                    current_delta = np.dot(self.weights[layer][:, 1:].T, current_delta) * activation_derivative[1:]
                    weight_gradients[layer - 1] += np.dot(current_delta.reshape(-1, 1), activations[layer - 1].T)
            weight_gradient_array = [1 / self.number_of_samples * gradient for gradient in weight_gradients]

            for i, gradient in enumerate(weight_gradient_array):
                gradient[1:, :] = gradient[1:, :] + self.regularization_coefficient / self.number_of_samples * self.weights[i][1:, :]

            self.weights = [self.weights[i] - self.learning_rate * weight_gradient_array[i] for i in range(number_of_layers)]

            cost += (self.regularization_coefficient / (2 * self.number_of_samples)) * sum([np.sum(weight ** 2) for weight in self.weights])
            if iteration == self.training_iterations - 1 or np.mod(iteration, 10) == 0:
                print('Cost function cost = ', cost[0], 'in iteration',
                      iteration , 'with regularization_coefficient = ', self.regularization_coefficient)
                iteration , accuracy = self.ff_predict()
                # accuracy = np.sum(predicted_labels==target_labels) / m *100
                print('Net accuracy for training set = ', accuracy)

        return cost, self.weights




    def ff_predict():
        """
        ff_predict employs forward propagation on a 3 layer networks and
        determines the labels of  the inputs
        Input arguments
        Theta1 - matrix of parameters (weights)  between the input and the first hidden layer
        Theta2 - matrix of parameters (weights)  between the hidden layer and the output layer (or
            another hidden layer)
        X - input matrix
        y - input labels
        Output arguments:
        p - the predicted labels of the inputs
        Usage: p = ff_predict(Theta1, Theta2, X)
        """

        m = self.input_data.shape[0]
        p = np.zeros((m, 1))

        a = X
        i = 0
        for Theta in self.weights:
            ones = np.ones((a.shape[0], 1))
            a = np.concatenate((ones, a), axis=1)
            z = np.dot(a, Theta.T)
            a = self.activation_functions[i](z)
            i += 1
        p = np.argmax(a.T, axis=0)
        p = p.reshape(p.shape[0], 1)
        detectp = np.sum(p == y) / m * 100

        return p, detectp

