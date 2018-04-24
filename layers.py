from abc import ABC
import numpy as np
import functions as fn


class Layer(ABC):
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.outputs = None
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
        self.grad_x_local = None
        self.grad_w_local = None
        self.total_w_grad = None
        self.total_x_grad = None


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
        self.grad_x_local = None
        self.grad_w_local = None
        self.total_w_grad = None
        self.total_x_grad = None

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

    def update_parameters(self, learning_rate = 0.0001):
        self.weights -= learning_rate * self.total_w_grad
        self.bias -= learning_rate * self.total_grad_bias


class SoftMax(Layer):
    def __init__(self, input_size):
        self.output_size = input_size
        super().__init__(input_size, self.output_size)
        self.weights = 0.01 * np.random.randn(self.output_size, input_size)
        self.bias = np.zeros((self.output_size, 1))
        self.outputs = None
        self.local_grad = None
        self.inputs = None
        self.scalar = None
        self.grad_scalar = None
        self.sum = None
        self.grad_bias_local = None
        self.total_grad_bias = None
        self.grad_x_local = None
        self.grad_w_local = None
        self.total_w_grad = None
        self.total_x_grad = None

    def compute(self, inputs):
        self.outputs = fn.softmax(inputs)
        return self.outputs

    def get_local_grad_matrix(self):
        q_rows = np.repeat(self.outputs.transpose(), self.output_size, 0)
        identity = np.identity(self.output_size)
        self.grad_x_local = self.outputs * (identity - q_rows)
        return self.grad_x_local

    def back_compute(self, upstream):
        self.get_local_grad_matrix()
        self.total_x_grad = np.dot(np.transpose(self.grad_x_local), upstream)
        return self.total_x_grad

    def update_parameters(self, learning_rate = 0.0001):
        pass


class SquareLoss(Layer):
    def __init__(self):
        pass


class CrossEntropyLoss(Layer):
    def __init__(self, input_size):
        self.output_size = input_size
        super().__init__(input_size, self.output_size)
        self.weights = 0.01 * np.random.randn(self.output_size, input_size)
        self.bias = np.zeros((self.output_size, 1))
        self.outputs = None
        self.local_grad = None
        self.inputs = None
        self.scalar = None
        self.grad_scalar = None
        self.sum = None
        self.grad_bias_local = None
        self.total_grad_bias = None
        self.grad_x_local = None
        self.grad_w_local = None
        self.total_w_grad = None
        self.total_x_grad = None

    def compute(self, inputs, actual):
        self.inputs = inputs
        logs = np.log(inputs)
        self.outputs = -np.sum(actual * logs)
        return self.outputs

    def get_local_grad_matrix(self):
        self.grad_x_local = 1 / self.inputs
        return self.grad_x_local

    def back_compute(self, actual):
        self.total_x_grad = -actual / self.inputs
        return self.total_x_grad

    def update_parameters(self, learning_rate = 0.0001):
        pass