import numpy as np
from autograd import grad


class Module:
    def __init__(self):
        self.parameters = list()

    def __call__(self, x):
        self.forward(x)

    def forward(self, x): raise NotImplementedError

    
class Linear(Module):
    def __init__(self, in_features, out_features):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # parameters
        self.weights = np.random.uniform(-1, 1, (out_features, in_features))
        self.bias = np.random.uniform(-1, 1, (out_features,))

        self.parameters += [self.weights, self.bias]

    def forward(self, x):
        if len(x) != 1: raise NameError('only one tensor allowed as input')
        x = x[0]
        return np.add(np.dot(x, self.weights.T), self.bias)

class Tanh(Module):
    def __init__(self):
        super(Tanh, self).__init__()

    def forward(self, x):
        return 1./(1 + np.exp(-x))








        