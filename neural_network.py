from typing import List, Callable, Any
from matplotlib import pyplot as plt
import numpy as np
from utils import init_parameters

'''
This file contains the implementation of a Neural Network model.
'''


class model:
    def __init__(self, hidden_layer_dimensions: List[int], input_data: np.ndarray, target_labels: np.ndarray,
                 activation_functions: List[Callable], derivative_functions: List[Callable],
                 training_iterations: int, learning_rate: float, regularization_coefficient: float) -> None:
        """
        Initialize the Neural Network model.
        :param hidden_layer_dimensions: A list of the number of neurons in each hidden layer.
        :param input_data: The input data matrix.
        :param target_labels: The labels of the input data.
        :param activation_functions: A list of the activation functions for each layer.
        :param derivative_functions: A list of the derivative functions for each layer.
        :param training_iterations: The number of training iterations.
        :param learning_rate: The learning rate.
        :param regularization_coefficient: The regularization coefficient.
        """
        output_dimensions = np.unique(target_labels).size
        self.layer_dimensions = [input_data.shape[1]] + hidden_layer_dimensions + [output_dimensions]
        self.weights = self.init_weights()
        self.input_data = input_data
        self.target_labels = target_labels
        self.activation_functions = activation_functions
        self.derivative_functions = derivative_functions
        self.training_iterations = training_iterations
        self.learning_rate = learning_rate
        self.regularization_coefficient = regularization_coefficient
        self.number_of_samples = input_data.shape[0]
        self.cost_history = np.zeros((self.training_iterations, 1))

    def init_weights(self) -> List[np.ndarray]:
        """
        Initialize the weights of the Neural Network using "Xavier Initialization".
        Xavier Initialization is a method of weight initialization that keeps the variance of the activations
        approximately equal across all layers, which helps to prevent the problem of vanishing/exploding gradients.

        :return: A list of the weights of the Neural Network.
        """
        weights_list = []
        for input_layer_dimension, output_layer_dimension in zip(self.layer_dimensions, self.layer_dimensions[1:]):
            weights_list.append(init_parameters(input_layer_dimension, output_layer_dimension))
        return weights_list

    def feed_forward(self, index: int) -> List[np.ndarray]:
        """
        Feed forward through the Neural Network.

        :param index: The index of the input data to feed forward.
        :return: A list of the activations of each layer.
        """
        activations = []
        input_data = self.input_data[index]
        activation = input_data.reshape(-1, 1)

        for i, weights in enumerate(self.weights):
            ones = np.ones((1, 1))
            activation = np.concatenate((ones, activation), axis=0)
            z = np.dot(weights, activation)
            activation = self.activation_functions[i](z)
            activations.append(activation)
        activations.insert(0, input_data)
        activations[:-1] = [np.insert(a, 0, 1) for a in activations[:-1]]  # add 1 bias
        activations = [a.reshape(a.shape[0], 1) for a in activations]
        return activations

    def get_hot_vector(self, predicted_labels: List[np.ndarray], index: int) -> np.ndarray:
        """
        Get the hot vector of the target label.

        :param predicted_labels: The predicted labels of the input data.
        :param index: The index of the input data.
        :return: The hot vector of the target label.
        """
        binary_target = np.zeros(predicted_labels[-1].shape)
        binary_target[self.target_labels[index]] = 1
        return binary_target

    @staticmethod
    def calculate_cost(predicted_labels: np.ndarray, hot_vector: np.ndarray) -> float:
        """
        Calculate the cost of the Neural Network.

        :param predicted_labels: The predicted labels of the input data.
        :param hot_vector: The hot vector of the target label.
        :return: The cost of the Neural Network.
        """
        epsilon = 1e-8
        cost = (-1) * (np.dot(hot_vector.T, np.log(np.maximum(predicted_labels, epsilon))) +
                       np.dot((1 - hot_vector).T, np.log(np.maximum(1 - predicted_labels, epsilon))))
        return cost

    def predict_output(self, input_data: np.ndarray, target_labels: np.ndarray) -> tuple[Any, float | int | Any]:
        """
        Predict the output of the Neural Network and calculate the accuracy.

        :return: The predicted output of the Neural Network.
        :return: The accuracy of the Neural Network.
        """
        m = input_data.shape[0]
        np.zeros((m, 1))
        activation = input_data

        for i, weight in enumerate(self.weights):
            ones = np.ones((activation.shape[0], 1))
            activation = np.concatenate((ones, activation), axis=1)
            z = np.dot(activation, weight.T)
            activation = self.activation_functions[i](z)

        predicted_labels = np.argmax(activation.T, axis=0)
        predicted_labels = predicted_labels.reshape(predicted_labels.shape[0], 1)
        accuracy = (np.sum(predicted_labels == target_labels) / m) * 100

        return predicted_labels, accuracy

    def print_network_configuration(self) -> None:
        """
        Print the configuration of the Neural Network.
        """
        print('Neural Network Configuration:\n')
        print('Number of training samples      = ', self.number_of_samples)
        print('Number of neurons in each layer = ', self.layer_dimensions)
        print('Activation functions            = ', [function.__name__ for function in self.activation_functions])
        print('Number of training iterations   = ', self.training_iterations)
        print('Learning rate                   = ', self.learning_rate)
        print('Regularization coefficient      = ', self.regularization_coefficient)
        print('Number of parameters            = ', sum([weight.size for weight in self.weights]))
        print('\n\n')

    def print_training_progress(self, iteration: int, cost: float) -> None:
        """
        Print the training progress of the Neural Network.

        :param iteration: The current iteration of the Neural Network training.
        :param cost: The cost of the Neural Network.
        """

        print('Training iteration:', iteration, 'Cost:', cost[0])
        _, accuracy = self.predict_output(self.input_data, self.target_labels)
        print('Training accuracy:', accuracy, '\n')

    def should_print_progress(self, iteration: int) -> bool:
        """
        :return: True if the iteration is the last iteration or a multiple of 10, False otherwise.
        """
        return iteration == self.training_iterations - 1 or np.mod(iteration, 10) == 0

    def back_propagation(self, current_delta: np.ndarray, predicted_labels: List[np.ndarray],
                         weight_gradients: List[np.ndarray]) -> None:
        """
        Backpropagate through the Neural Network.

        :param current_delta: The current delta of the Neural Network.
        :param predicted_labels: The predicted labels of the input data.
        :param weight_gradients: The weight gradients of the Neural Network.
        :return: The weight gradients of the Neural Network (no return value needed).
        """
        number_of_layers = len(self.activation_functions)
        for layer in range(number_of_layers - 1, 0, -1):
            derivative_of_activation = self.derivative_functions[layer - 1](predicted_labels[layer])
            current_delta = np.dot(self.weights[layer][:, 1:].T, current_delta) * derivative_of_activation[1:]
            weight_gradients[layer - 1] += np.dot(current_delta.reshape(-1, 1), predicted_labels[layer - 1].T)

    def train_neural_network(self):
        """
        Train the Neural Network.
        :return: The cost and the weights of the Neural Network.
        """
        print('\n\n-------------starting NN training-------------\n')
        self.print_network_configuration()

        print('\n--- printing training progress: \n')
        for iteration in range(self.training_iterations):
            weight_gradients = [np.zeros(weight.shape) for weight in self.weights]
            random_indices = np.random.permutation(self.number_of_samples)

            for index in random_indices:
                cost = 0
                activations_arr = self.feed_forward(index)

                ### start Backward propagation ###
                hot_vector = self.get_hot_vector(activations_arr, index)
                predicted_output = activations_arr[-1]
                cost += self.calculate_cost(predicted_output, hot_vector)

                # first the output layer
                output_layer_delta = (activations_arr[-1] - hot_vector)
                weight_gradients[-1] += np.dot(output_layer_delta, activations_arr[-2].T)

                # and now all the other layers
                self.back_propagation(output_layer_delta, activations_arr, weight_gradients)
            weight_gradient_array = [1 / self.number_of_samples * gradient for gradient in weight_gradients]

            # update weights and add regularization term
            regularization_term = self.regularization_coefficient / self.number_of_samples
            for i, gradient in enumerate(weight_gradient_array):
                gradient[1:, :] += regularization_term * self.weights[i][1:, :]
                self.weights[i] -= self.learning_rate * weight_gradient_array[i]

            # add regularization to cost function
            cost += (regularization_term / 2) * sum([np.sum(weight ** 2) for weight in self.weights])
            self.cost_history[iteration] = cost

            if self.should_print_progress(iteration):
                self.print_training_progress(iteration, cost)

        # plot_loss()
        return cost, self.weights

    def plot_loss(self) -> None:
        """
        Plot the loss of the Neural Network.
        """
        plt.plot(self.cost_history)
        plt.xlabel('Training Iterations')
        plt.ylabel('Cost')
        plt.title('Cost vs Training Iterations')
        plt.show()
