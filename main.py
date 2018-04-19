import matplotlib.pyplot as plt
import numpy as np

import functions as fn

small_scalar = 0.15
_learning_rate = 0.001


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
    # lets visualize the data:

    return X, y


@np.vectorize
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


def num_to_binary(integer):
    result = []
    while integer > 0:
        remainder = integer % 2
        result.append(remainder)
        integer -= remainder
        integer /= 2
    return result


@np.vectorize
def num_to_list(integer):
    result = [0 for _ in range(3)]
    result[integer] = 1
    return result


def visualise(X, y):
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
    plt.show()


class NeuralNet:
    def __init__(self, input_size):
        self.layers = []
        self.input_size = input_size

    @staticmethod
    def xavier_initialization(self):
        pass

    def forward_pass(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer.compute(x)
        return x

    def back_prop(self, actual):
        up_grads = np.array([[2], [2], [2]]) * (self.layers[-1].outputs - actual) #todo remove hardcode
        for layer in self.layers[::-1]:
            up_grads = layer.back_compute(up_grads)
            layer.update_parameters()

    @staticmethod
    def loss_squares(outputs, actual):
        def grad():
            return 2 * outputs

        return np.sum(np.abs(outputs - actual) ** 2)

    class Layer:
        def __init__(self, input_size, output_size):
            self.input_size = input_size
            self.output_size = output_size
            self.outputs = None

        def initialize(self, n, inputs=1):
            factor = small_scalar
            return factor * (np.random.randn(n, inputs) - 0.5)

    class ReLu(Layer):
        def __init__(self, input_size, output_size):
            super().__init__(input_size, output_size)
            self.weights = 0.01 * np.random.randn(output_size, input_size)
            self.bias = np.zeros((output_size, 1))
            self.outputs = None
            self.local_grad = None
            self.inputs = None
            self.scalar = None
            self.grad_scalar = None
            self.sum = None
            self.grad_bias_local = None
            self.total_grad_bias = None

        def get_local_grad_matrix(self, inputs, outputs):
            self.local_grad = self.weights

        def compute(self, inputs):
            self.sum = np.dot(self.weights, inputs) + self.bias
            self.grad_scalar = np.ones((self.output_size, 1))
            self.grad_scalar[
                self.sum <= 0] = 0  # the gradient for each output neuron wrt the function

            self.grad_x_local = self.weights * self.grad_scalar  # needs to be also multiplied by upstream

            self.grad_w_local = np.repeat(np.transpose(inputs), self.output_size, axis=0)

            self.outputs = fn.relu(np.dot(self.weights, inputs) + self.bias)
            return self.outputs

        def back_compute(self, upstream):
            self.total_w_grad = self.grad_w_local * upstream
            self.total_x_grad = np.dot(np.transpose(self.grad_x_local), upstream)
            self.total_grad_bias = self.grad_scalar * upstream
            return self.total_x_grad

        def update_parameters(self):
            self.weights -= _learning_rate * self.total_w_grad
            self.bias -= _learning_rate * self.total_grad_bias

    class SoftMax(Layer):
        def __init__(self, input_size, output_size):
            super().__init__(input_size, output_size)
            self.weights = self.initialize(self.output_size, self.input_size)
            self.bias = self.initialize(output_size, 1)
            self.outputs = None

        def compute(self, inputs):
            self.outputs = fn.softmax(np.dot(self.weights, inputs) + self.bias)
            return self.outputs

    def add_layer(self, kind='relu'):
        if kind == 'relu':
            self.add_relu()

    def add_relu(self, hidden_size):
        _input_size = self.layers[-1].output_size if len(self.layers) > 0 else self.input_size

        self.layers.append(self.ReLu(_input_size, hidden_size))

    @staticmethod
    def initialize(n, inputs=1):
        factor = small_scalar
        return factor * (np.random.randn(n, inputs) - 0.5)


inputs = 100 * np.array([[100, 200]])

X, y = generate_data()

X_train = X[:300]
X_train = [np.reshape(x, (2, 1)) for x in X_train]
X_test = X[200:]
X_test = [np.reshape(x, (2, 1)) for x in X_test]

Y_train = y[:300]
Y_train = [np.reshape(num_to_list(z), (3, 1)) for z in Y_train]
Y_test = y[200:]
Y_test = [np.reshape(num_to_list(z), (3,1)) for z in Y_test]
train_data = list(zip(X_train, Y_train))
test_data = list(zip(X_test, Y_test))
visualise(X, y)


def update():
    pass
