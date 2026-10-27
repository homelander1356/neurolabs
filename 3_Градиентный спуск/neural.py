import numpy as np

class Perceptron:
    def __init__(self, inputSize, hiddenSizes, outputSize):
        
        self.Win = np.zeros((1+inputSize,hiddenSizes))
        self.Win[0,:] = (np.random.randint(0, 3, size = (hiddenSizes)))
        self.Win[1:,:] = (np.random.randint(-1, 2, size = (inputSize,hiddenSizes)))
        
        self.Wout = np.random.randint(0, 2, size = (1+hiddenSizes,outputSize)).astype(np.float64)
        
    def predict(self, Xp):
        hidden_predict = np.where((np.dot(Xp, self.Win[1:,:]) + self.Win[0,:]) >= 0.0, 1, -1).astype(np.float64)
        out = np.where((np.dot(hidden_predict, self.Wout[1:,:]) + self.Wout[0,:]) >= 0.0, 1, -1).astype(np.float64)
        return out, hidden_predict

    def train(self, X, y, n_iter=5, eta = 0.01):
        for i in range(n_iter):
            print(self.Wout.reshape(1, -1))
            for xi, target, j in zip(X, y, range(X.shape[0])):
                pr, hidden = self.predict(xi)
                self.Wout[1:] += ((eta * (target - pr)) * hidden).reshape(-1, 1)
                self.Wout[0] += eta * (target - pr)
        return self

class MLP:
    def __init__(self, inputSize, outputSize, learning_rate=0.1, hiddenSizes=5):
        self.weights = [
            np.random.uniform(-2, 2, size=(inputSize, hiddenSizes)),
            np.random.uniform(-2, 2, size=(hiddenSizes, outputSize))
        ]
        self.learning_rate = learning_rate
        self.layers = None

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def feed_forward(self, x):
        input_ = np.atleast_2d(x)  # (1, inputSize)
        hidden_ = self.sigmoid(np.dot(input_, self.weights[0]))   # (1, hidden)
        output_ = self.sigmoid(np.dot(hidden_, self.weights[1]))  # (1, output)

        self.layers = [input_, hidden_, output_]
        return self.layers[-1]

    def backward(self, target):
        target = np.atleast_2d(target)  # (1, output)
        err = target - self.layers[-1]

        for i in range(len(self.layers) - 1, 0, -1):
            err_delta = err * (self.layers[i] * (1 - self.layers[i]))
            err = np.dot(err_delta, self.weights[i - 1].T)
            dw = np.dot(self.layers[i - 1].T, err_delta)
            self.weights[i - 1] += self.learning_rate * dw

    def train(self, x_values, target):
        idx = np.random.permutation(len(x_values))
        for i in idx:
            self.feed_forward(x_values[i])
            self.backward(target[i])

    def predict(self, x_values):
        x_values = np.atleast_2d(x_values)
        hidden_ = self.sigmoid(np.dot(x_values, self.weights[0]))
        output_ = self.sigmoid(np.dot(hidden_, self.weights[1]))
        return output_