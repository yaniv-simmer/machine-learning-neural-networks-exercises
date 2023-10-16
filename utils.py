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


def backprop(Theta_arr, X, y, activation_arr, max_iter=1000, alpha=0.9, Lambda=0.0):
    """
    backprop - BackPropagation for training a neural network
    Input arguments
    Theta1 - matrix of parameters (weights)  between the input and the first
        hidden layer
    Theta2 - matrix of parameters (weights)  between the hidden layer and the
        output layer (or another hidden layer)
    X - input matrix
    y - labels of the input examples
    max_iter - maximum number of iterations (epochs).
    alpha - learning coefficient.
    Lambda - regularization coefficient.

    Output arguments
    J - the cost function
    Theta1 - updated weight matrix between the input and the first
        hidden layer
    Theta2 - updated weight matrix between the hidden layer and the output
        layer (or a second hidden layer)

    Usage:
    [J,Theta1,Theta2] = backprop(Theta1, Theta2, X,y,max_iter, alpha,Lambda)
    """
    m = X.shape[0]
    num_outputs = Theta_arr[-1].shape[0]
    p = np.zeros((m, 1))
    J = 0
    epsilon = 1e-8

    for q in range(max_iter):
        dThetas = [np.zeros(theta.shape) for theta in Theta_arr]
        r = np.random.permutation(m)
        layers_num = len(activation_arr)

        for k in r:

            J = 0
            a = feed_forward(Theta_arr, X[k], activation_arr)
            ### start Backward propagation

            # Assigning 1 to the binary digit according to the class (label) of the input
            ybin = np.zeros(a[-1].shape)
            ybin[y[k]] = 1

            J += (-1) * (np.dot(ybin.T, np.log(np.maximum(a[-1], epsilon))) +
                         np.dot((1 - ybin).T, np.log(np.maximum(1 - a[-1], epsilon))))
            # output layer
            curr_delta = (a[-1] - ybin)
            dThetas[-1] += np.dot(curr_delta, a[-2].T)

            for l in range(layers_num - 1, 0, -1):
                g_tag = activation_arr[l](a[l])
                curr_delta = np.dot(Theta_arr[l][:, 1:].T, curr_delta) * g_tag[1:]
                dThetas[l - 1] += np.dot(curr_delta.reshape(-1, 1), a[l - 1].T)
        Theta_grad_arr = [1 / m * d for d in dThetas]

        for i, grad in enumerate(Theta_grad_arr):
            grad[1:, :] = grad[1:, :] + Lambda / m * Theta_arr[i][1:, :]

        Theta_arr = [Theta_arr[i] - alpha * Theta_grad_arr[i] for i in range(layers_num)]

        J += (Lambda / (2 * m)) * sum([np.sum(theta ** 2) for theta in Theta_arr])
        if q == max_iter - 1 or np.mod(q, 10) == 0:
            print('Cost function J = ', J[0], 'in iteration',
                  q, 'with Lambda = ', Lambda)
            p, acc = ff_predict(Theta_arr, activation_arr, X, y)
            # acc = np.sum(p==y) / m *100
            print('Net accuracy for training set = ', acc)
            # time.sleep(0.005)

    return J, Theta_arr
