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

