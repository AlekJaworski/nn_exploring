import numpy as np


def relu(y):
    return np.maximum(y, 0)


def softmax(y):
    total = np.sum(np.e ** y)
    return np.e ** y / total


def sigmoid(y):
    return 1 / (1 + np.e ** -y)