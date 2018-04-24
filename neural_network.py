import numpy as np
import layers


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
