from abc import ABC, abstractmethod

import numpy as np


class Function(ABC):
    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def result(self, *args):
        pass

    @abstractmethod
    def gradient(self, *args):
        pass

class CrossEntropyLoss(Function):
    def __init__(self, outputs, actual):
        self.inputs = outputs
        self.actual = actual
        self.grad = None
        self.outputs = None

    def result(self):
        self.outputs = -np.dot(self.actual, np.log(self.inputs))
        return self.outputs

    def gradient(self):
        total = np.sum(self.inputs)
        self.grad = self.inputs - np.ones(len(self.actual)) * self.actual
        return self.grad

class LossSquares(Function):
    def __init__(self, outputs, actual):
        self.inputs = outputs
        self.actual = actual
        self.grad = None
        self.outputs = None

    def result(self):
        differences = self.actual - self.inputs
        self.outputs = 0.5*np.dot(differences, differences)
        return self.outputs

    def gradient(self):
        self.grad = self.inputs
        return self.grad

def relu(y):
    return np.maximum(y, 0)


def softmax(y):
    total = np.sum(np.e ** y)
    return np.e ** y / total


def sigmoid(y):
    return 1 / (1 + np.e ** -y)

class Sigmoid(Function):

    def __init__(self, inputs):
        self.inputs = inputs
        self.grad = None
        self.outputs = None

    def result(self, inputs):
        self.outputs = 1/(1 - np.e ** inputs)

    def gradient(self, *args):
        if self.outputs is None:
            self.result()
        self.grad = self.outputs * (np.ones(len(self.outputs)) - self.outputs)
        return self.grad


class RELU(Function):
    def __init__(self, inputs):
        self.inputs = inputs
        self.grad = None
        self.local_grad = None
        self.outputs = None

    def result(self):
        self.outputs = relu(self.inputs)
        return self.outputs

    def gradient(self):
        pred = np.vectorize(lambda x: 1 if x > 0 else 0)
        self.grad = pred(self.inputs)
        return self.grad

class SVM(Function):
    def __init__(self, inputs, weights):
        self.inputs = inputs
        self.weights = weights
        self.grad = None
        self.outputs = None
        self.local_grad = None
        self.bias = np.random.randn(1, 1)

    def result(self):
        self.outputs = np.dot(self.inputs, self.weights)
        return self.outputs

    def gradient(self, upstream):
        self.grad = upstream * self.local_gradient()
        return self.grad

    def update_weights(self, learning_rate, upstream):
        if self.grad is None:
            self.gradient(upstream)
        self.weights -= self.grad*learning_rate

    def x_gradient(self, upstream):
        self.downstream_grad = upstream * self.inputs

    def local_gradient(self):
        self.local_grad = self.inputs
        return self.local_grad
