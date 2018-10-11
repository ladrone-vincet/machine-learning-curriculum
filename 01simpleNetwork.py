#https://towardsdatascience.com/how-to-build-your-own-neural-network-from-scratch-in-python-68998a08e4f6
#Triple XOR
import numpy as np


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1.0 - x)


class SimpleNetwork:

    def __init__(self, x, y):
        self.input      = x
        self.weigths1   = np.random.rand(self.input.shape[1], 4)
        self.weigths2   = np.random.rand(4, 1)
        self.y          = y
        self.output     = np.zeros(y.shape)

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weigths1))
        self.output = sigmoid(np.dot(self.layer1, self.weigths2))

    def backpropagation(self):
        d_weights2 = np.dot(self.layer1.T, (2 * (self.y - self.output) * sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T,
                            np.dot(2 * (self.y - self.output) * sigmoid_derivative(self.output), self.weigths2.T) * sigmoid_derivative(self.layer1))

        self.weigths1 += d_weights1
        self.weigths2 += d_weights2


if __name__ == "__main__":
    X = np.array([
        [0, 0, 1],
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 1]])

    y = np.array([
        [0],
        [1],
        [1],
        [0]])

    sn = SimpleNetwork(X, y)

    for i in range(4000):
        sn.feedforward()
        sn.backpropagation()
        print(sn.output)
