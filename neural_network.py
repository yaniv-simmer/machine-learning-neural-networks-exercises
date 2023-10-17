from utils import init_parameters
from typing import List, Callable
import matplotlib.pyplot as plt
import numpy as np


class model:
    def __init__(self, hidden_layer_dimensions: List[int], input_data: np.ndarray, target_labels: np.ndarray,
                 activation_functions: List[Callable], derivative_functions: List[Callable],
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
        self.derivative_functions = derivative_functions
        self.training_iterations = training_iterations
        self.learning_rate = learning_rate
        self.regularization_coefficient = regularization_coefficient
        self.number_of_samples = input_data.shape[0]



    def feed_forward(self, index: int)->List[np.ndarray]:
        a_arr = []
        X = self.input_data[index]
        a0 = X.reshape(-1, 1)
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



    def get_hot_vector(self, predicted_labels: np.ndarray, index: int)->np.ndarray:
        # Assigning 1 to the binary digit according to the class (label) of the input
        binary_target = np.zeros(predicted_labels[-1].shape)
        binary_target[self.target_labels[index]] = 1
        return binary_target



    def calculate_cost(self, predicted_labels: np.ndarray, hot_vector: np.ndarray, index: int)->float:
        ellipsis = 1e-8
        cost = (-1) * (np.dot(hot_vector.T, np.log(np.maximum(predicted_labels, ellipsis))) +
                          np.dot((1 - hot_vector).T, np.log(np.maximum(1 - predicted_labels, ellipsis))))
        return cost




    def ff_predict(self):


        m = self.input_data.shape[0]
        p = np.zeros((m, 1))

        a = self.input_data
        i = 0
        for Theta in self.weights:
            ones = np.ones((a.shape[0], 1))
            a = np.concatenate((ones, a), axis=1)
            z = np.dot(a, Theta.T)
            a = self.activation_functions[i](z)
            i += 1
        p = np.argmax(a.T, axis=0)
        p = p.reshape(p.shape[0], 1)
        detectp = np.sum(p == self.target_labels) / m * 100

        return p, detectp








    def train(self):
        """
        Train the Neural Network using backpropagation.

        """
        # ??? predicted_labels = np.zeros((self.number_of_samples, 1))
        cost_history = np.zeros((self.training_iterations, 1))
        

        for iteration  in range(self.training_iterations):
            weight_gradients = [np.zeros(weight.shape) for weight in self.weights]
            random_indices = np.random.permutation(self.number_of_samples)
            number_of_layers = len(self.activation_functions)
            
            for index in random_indices:
                cost = 0
                predicted_labels = self.feed_forward(index)
                
                ### start Backward propagation
                hot_vector = self.get_hot_vector(predicted_labels, index)
                cost += self.calculate_cost(predicted_labels[-1], hot_vector, index)
                
                # first the output layer
                current_delta = (predicted_labels[-1] - hot_vector)
                weight_gradients[-1] += np.dot(current_delta, predicted_labels[-2].T)
                
                # and now all the other layers
                for layer in range(number_of_layers - 1, 0, -1):
                    derivative_of_activation = self.derivative_functions[layer-1](predicted_labels[layer])
                    current_delta = np.dot(self.weights[layer][:, 1:].T, current_delta) * derivative_of_activation[1:]
                    weight_gradients[layer - 1] += np.dot(current_delta.reshape(-1, 1), predicted_labels[layer - 1].T)
            weight_gradient_array = [1 / self.number_of_samples * gradient for gradient in weight_gradients]

            for i, gradient in enumerate(weight_gradient_array):
                gradient[1:, :] = gradient[1:, :] + self.regularization_coefficient / self.number_of_samples * self.weights[i][1:, :]

            self.weights = [self.weights[i] - self.learning_rate * weight_gradient_array[i] for i in range(number_of_layers)]

            cost += (self.regularization_coefficient / (2 * self.number_of_samples)) * sum([np.sum(weight ** 2) for weight in self.weights])
            cost_history[iteration] = cost
            if iteration == self.training_iterations - 1 or np.mod(iteration, 10) == 0:
                print('Cost function cost = ', cost[0], 'in iteration',
                      iteration , 'with regularization_coefficient = ', self.regularization_coefficient)
                iteration , accuracy = self.ff_predict()
                # accuracy = np.sum(predicted_labels==target_labels) / m *100
                print('Net accuracy for training set = ', accuracy)
        plt.plot(cost_history)
        plt.show()
        return cost, self.weights



