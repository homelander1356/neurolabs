import numpy as np

class Perceptron:
    def __init__(self, inputSize, hiddenSizes, outputSize):
        
        # веса от входа к первому скрытому слою
        self.Win = np.zeros((1+inputSize,hiddenSizes))
        self.Win[0,:] = (np.random.randint(0, 3, size = (hiddenSizes)))
        self.Win[1:,:] = (np.random.randint(-1, 2, size = (inputSize,hiddenSizes)))
        
        # веса между скрытыми слоями
        self.Whid = np.zeros((1+hiddenSizes,hiddenSizes))
        self.Whid[0,:] = (np.random.randint(0, 3, size = (hiddenSizes)))
        self.Whid[1:,:] = (np.random.randint(-1, 2, size = (hiddenSizes,hiddenSizes)))
        
        # веса выходного слоя
        self.Wout = np.random.randint(0, 2, size = (1+hiddenSizes,outputSize)).astype(np.float64)
        #self.Wout = np.random.randint(0, 3, size = (1+hiddenSizes,outputSize))
        
    def predict(self, Xp):
        net1 = np.dot(Xp, self.Win[1:, :]) + self.Win[0, :]
        hidden1 = np.where(net1 >= 0.0, 1, -1).astype(np.float64)
        
        net2 = np.dot(hidden1, self.Whid[1:, :]) + self.Whid[0, :]
        hidden2 = np.where(net2 >= 0.0, 1, -1).astype(np.float64)
        
        out_net = np.dot(hidden2, self.Wout[1:, :]) + self.Wout[0, :]
        out = np.where(out_net >= 0.0, 1, -1).astype(np.float64)
        
        return out, hidden2

    def train(self, X, y, n_iter=5, eta = 0.01):
        for i in range(n_iter):
            print(self.Wout.reshape(1, -1))
            for xi, target, j in zip(X, y, range(X.shape[0])):
                pr, hidden = self.predict(xi)
                self.Wout[1:] += ((eta * (target - pr)) * hidden).reshape(-1, 1)
                self.Wout[0] += eta * (target - pr)
        return self

