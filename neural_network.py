from typing import List, Callable

import numpy as np

from utils import init_parameters, ff_predict, feed_forward


class model:
    def __init__(self, hidden_dim: List[int], X: np.ndarray, y: np.ndarray,
                 activations: List[Callable],
                 epochs: int, alpha: float, Lambda: float):
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
        outputs_dim = np.unique(y).size
        self.layers_dim = [X.shape[1]] + hidden_dim + [outputs_dim]
        self.W = [init_parameters(in_dim, out_dim) for in_dim, out_dim in zip(self.layers_dim, self.layers_dim[1:])]
        self.X = X
        self.y = y
        self.activations = activations
        self.epochs = epochs
        self.alpha = alpha
        self.Lambda = Lambda
        self.m = X.shape[0]

    def train(self):
        """
        Train the Neural Network using backpropagation.

        """
        # J, Theta_arr = backprop(self.W, self.X, self.y, self.activations, self.epochs, self.alpha, self.Lambda)
        # Theta_arr = self.W
        num_outputs = self.W[-1].shape[0]
        p = np.zeros((self.m, 1))
        J = 0
        epsilon = 1e-8

        for q in range(self.epochs):
            dWehits = [np.zeros(w.shape) for w in self.W]
            r = np.random.permutation(self.m)
            layers_num = len(self.activations)

            for k in r:

                J = 0
                a = feed_forward(self.W, self.X[k], self.activations)
                ### start Backward propagation

                # Assigning 1 to the binary digit according to the class (label) of the input
                ybin = np.zeros(a[-1].shape)
                ybin[self.y[k]] = 1

                J += (-1) * (np.dot(ybin.T, np.log(np.maximum(a[-1], epsilon))) +
                             np.dot((1 - ybin).T, np.log(np.maximum(1 - a[-1], epsilon))))
                # output layer
                curr_delta = (a[-1] - ybin)
                dWehits[-1] += np.dot(curr_delta, a[-2].T)

                for l in range(layers_num - 1, 0, -1):
                    g_tag = self.activations[l](a[l])
                    curr_delta = np.dot(self.W[l][:, 1:].T, curr_delta) * g_tag[1:]
                    dWehits[l - 1] += np.dot(curr_delta.reshape(-1, 1), a[l - 1].T)
            Theta_grad_arr = [1 / self.m * d for d in dWehits]

            for i, grad in enumerate(Theta_grad_arr):
                grad[1:, :] = grad[1:, :] + self.Lambda / self.m * self.W[i][1:, :]

            self.W = [self.W[i] - self.alpha * Theta_grad_arr[i] for i in range(layers_num)]

            J += (self.Lambda / (2 * self.m)) * sum([np.sum(theta ** 2) for theta in self.W])
            if q == self.epochs - 1 or np.mod(q, 10) == 0:
                print('Cost function J = ', J[0], 'in iteration',
                      q, 'with Lambda = ', self.Lambda)
                p, acc = ff_predict(self.W, self.activations, self.X, self.y)
                # acc = np.sum(p==y) / m *100
                print('Net accuracy for training set = ', acc)
                # time.sleep(0.005)

        return J, self.W

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the labels of the input data X.
        :param X: The input data.
        :return: The predicted labels.
        """
        p, _ = ff_predict(self.W, self.activations, X)
        return p
