import matplotlib.pyplot as plt
import numpy as np

import layers as layers
import utils as utils

small_scalar = 0.15
_learning_rate = 0.001


class NeuralNet:
    def __init__(self, input_size):
        self.layers = []
        self.input_size = input_size
        self.loss_layer = None
        self.loss = None

    def add_cross_entropy_loss(self):
        self.loss_layer = layers.CrossEntropyLoss(self.layers[-1].output_size)

    @staticmethod
    def xavier_initialization(self):
        pass

    def forward_pass(self, inputs, actual):
        x = inputs
        for layer in self.layers:
            x = layer.compute(x)
        self.compute_loss(actual)
        # print(f'loss: {self.loss}')
        return x

    def compute_loss(self, actual):
        inputs = self.layers[-1].outputs

        self.loss = self.loss_layer.compute(inputs, actual)
        return self.loss

    def back_prop(self, actual, learning_rate):
        up_grads = self.loss_layer.back_compute(actual)

        for layer in self.layers[::-1]:
            up_grads = layer.back_compute(up_grads)
            layer.update_parameters(learning_rate)

    @staticmethod
    def loss_squares(outputs, actual):
        def grad():
            return 2 * outputs

        return np.sum(np.abs(outputs - actual) ** 2)

    def add_layer(self, kind='relu'):
        if kind == 'relu':
            self.add_relu()

    def add_relu(self, hidden_size):
        _input_size = self.layers[-1].output_size if len(self.layers) > 0 else self.input_size

        self.layers.append(layers.ReLu(_input_size, hidden_size))

    def add_softmax(self):
        _input_size = self.layers[-1].output_size if len(self.layers) > 0 else self.input_size
        self.layers.append(layers.SoftMax(_input_size))


inputs = 100 * np.array([[100, 200]])

X, y = utils.generate_data()

X_train = X[:300]
X_train = [np.reshape(x, (2, 1)) for x in X_train]
X_test = X[200:]
X_test = [np.reshape(x, (2, 1)) for x in X_test]

Y_train = y[:300]
Y_train = [np.reshape(utils.num_to_list(z), (3, 1)) for z in Y_train]
Y_test = y[200:]
Y_test = [np.reshape(utils.num_to_list(z), (3, 1)) for z in Y_test]
train_data = list(zip(X_train, Y_train))
test_data = list(zip(X_test, Y_test))
utils.visualise(X, y)


def visualise_boundary(net, granularity):
    granularity = granularity
    x = np.linspace(-1.5, 1.5, granularity)
    y = np.linspace(-1.5, 1.5, granularity)
    xv, yv = np.meshgrid(x, y)
    Z = np.zeros((granularity, granularity))
    good = 0
    for i in range(granularity):
        for j in range(granularity):
            Z[i, j] = utils.list_to_num(
                np.round(net.forward_pass(np.array([[xv[i, j]], [yv[i, j]]]), y.transpose())))

    plt.scatter(xv, yv, c=Z, s=40, cmap=plt.cm.Spectral)


    # plt.scatter(xv, yv, Z)
    plt.show()


neur_ex = NeuralNet(2)
neur_ex.add_relu(100)
neur_ex.add_relu(3)





def update():
    pass
