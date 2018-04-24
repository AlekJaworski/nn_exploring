import numpy as np
from matplotlib import pyplot as plt

from main import NeuralNet


def generate_data():
    N = 100  # number of points per class
    D = 2  # dimensionality
    K = 3  # number of classes
    X = np.zeros((N * K, D))  # data matrix (each row = single example)
    y = np.zeros(N * K, dtype='uint8')  # class labels
    for j in range(K):
        ix = range(N * j, N * (j + 1))
        r = np.linspace(0.0, 1, N)  # radius
        t = np.linspace(j * 4, (j + 1) * 4, N) + np.random.randn(N) * 0.2  # theta
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = j
    return X, y


def list_to_num(list):
    result = 0
    for index, number in enumerate(list):
        result += index * number
    return result


@np.vectorize
def num_to_list_padded(integer, padding):
    result = [0 for _ in range(padding)]
    small_form = num_to_list(integer)
    result[0:len(small_form)] = small_form
    return result


@np.vectorize
def num_to_list(integer):
    result = [0 for _ in range(3)]
    result[integer] = 1
    return result


def initialize_new():
    result = NeuralNet(2)
    result.add_relu(100)
    result.add_relu(3)
    result.add_softmax()
    result.add_cross_entropy_loss()
    return result


def visualise(X, y):
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
    plt.show()


def train_ex(net, data, iters, learning_rate, initial_iteration=0):
    for i in range(iters):
        _learning_rate = learning_rate / (1 + i + initial_iteration)
        if i % 100 == 99:
            print(f'iteration: {i}')
            print(f'learning rate: {_learning_rate}')
            print(f'loss: {net.loss}')
        for s in data:
            net.forward_pass(s[0], s[1])
            net.back_prop(s[1], learning_rate=_learning_rate)
