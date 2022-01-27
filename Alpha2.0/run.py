from module import Module, Linear, Tanh
from tensor import Parameter, Tensor
from autograd import grad
from autograd.misc.optimizers import adam
import numpy as np
class Net(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = Linear(in_features, out_features)
        self.act = Tanh()
     #   self.linear2 = Linear(out_features, out_features)
     #   self.act2 = Tanh()
     #   self.parame = Parameter([0,0,0,0,0])

    def forward(self, x):
        x = self.linear(x)
        x = self.act(x)
     #   x = self.linear2(x)
     #   x = self.act2(x)
        return x

if __name__ == '__main__':
    pass