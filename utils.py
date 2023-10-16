import numpy as np


def tanh(Z: np.ndarray) -> np.ndarray:
    """
    Compute the hyperbolic tangent of Z
    """
    A = (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))
    return A


def dtanh(x: np.ndarray) -> np.ndarray:
    """
    Compute the derivative of tanh(x)
    """
    tanh_x = np.tanh(x)
    dtanh_x = 1 - tanh_x ** 2
    return dtanh_x


def sigmoid(Z: np.ndarray) -> np.ndarray:
    """
    Compute the sigmoid of Z
    """
    A = 1 / (1 + np.exp(-Z))
    return A


def d_sigmoid(Z: np.ndarray) -> np.ndarray:
    """
    Compute the derivative of sigmoid(Z)
    """
    return Z * (1 - Z)


def ReLU(Z: np.ndarray) -> np.ndarray:
    """
    Compute ReLU(Z)
    """
    A = np.maximum(0, Z)
    return A


def dReLU(Z: np.ndarray) -> np.ndarray:
    """
    Compute dReLU(Z)
    """
    A = (Z > 0) * 1
    return A


def init_parameters(Lin: int, Lout: int) -> np.ndarray:
    """
    Randomly initialize the parameters of a layer with Lin incoming inputs and Lout outputs
    """
    factor = np.sqrt(6 / (Lin + Lout))
    Theta = np.zeros((Lout, Lin + 1))
    Theta = 2 * factor * (np.random.rand(Lout, Lin + 1) - 0.5)
    return Theta


def ff_predict(Theta_arr, activation_arr, X, y):
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

    m = X.shape[0]
    p = np.zeros((m, 1))

    a = X
    i = 0
    for Theta in Theta_arr:
        ones = np.ones((a.shape[0], 1))
        a = np.concatenate((ones, a), axis=1)
        z = np.dot(a, Theta.T)
        a = activation_arr[i](z)
        i += 1
    p = np.argmax(a.T, axis=0)
    p = p.reshape(p.shape[0], 1)
    detectp = np.sum(p == y) / m * 100

    return p, detectp


def feed_forward(Theta_arr, X, activation_arr):
    a_arr = []
    a0 = X.reshape(-1, 1)
    i = 0
    for Theta in Theta_arr:
        ones = np.ones((1, 1))
        a0 = np.concatenate((ones, a0), axis=0)
        z = np.dot(Theta, a0)

        a0 = activation_arr[i](z)
        a_arr.append(a0)
        i += 1
    a_arr.insert(0, X)
    a_arr[:-1] = [np.insert(a_, 0, 1) for a_ in a_arr[:-1]]  ## add 1 bias
    a_arr = [a_.reshape(a_.shape[0], 1) for a_ in a_arr]
    return a_arr


