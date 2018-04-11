import matplotlib.pyplot as plt
import numpy as np
import functions as fn

small_scalar = 0.15

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


def visualise(X, y):
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
    plt.show()

class NeuralNet:
    def __init__(self):
        self.layers = []
        self.input_size = 2
        self.output_size = 3


        self.W1 = self.initialize(2, 1)
        self.W2 = self.initialize(2, 6)
        self.W3 = self.initialize(6, 3)
        self.b1 = self.initialize(1, 1)
        self.b2 = self.initialize(1, 1)
        #self.h1 = self.relu(np.dot(x, W2) + b1)

    def forward_pass(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer.compute(x)
        return x

    class Layer:
        def __init__(self, input_size, output_size):
            self.input_size = input_size
            self.output_size = output_size

        def initialize(self, n, inputs=1):
            factor = small_scalar
            return factor * (np.random.randn(n, inputs) - 0.5)

    class ReLu(Layer):
        def __init__(self, input_size, output_size):
            super().__init__(input_size, output_size)
            self.weights = self.initialize(self.input_size, self.output_size)
            self.bias = self.initialize(1, 1)

        def compute(self, inputs):
            return fn.relu(np.dot(inputs, self.weights) + self.bias)

    class SoftMax(Layer):
        def __init__(self, input_size, output_size):
            super().__init__(input_size, output_size)
            self.weights = self.initialize(self.input_size, self.output_size)
            self.bias = self.initialize(1, 1)

        def compute(self, inputs):
            return fn.softmax(np.dot(inputs, self.weights) + self.bias)

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

nn = NeuralNet()
nn.add_relu(6)
nn.add_relu(4)


#out = softmax(np.dot(h1, W3) + b2)

X, y = generate_data()
visualise(X, y)


def update():
    pass
