from module import Tanh, Linear, Module
from tensor import Parameter, Tensor
import numpy as np
class Net(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = Linear(in_features, out_features)
        self.act = Tanh()
        self.parame = Parameter([0,0,0,0,0])

    def forward(self, x):
        x = self.linear(x)
        x = self.act(x)
        return x

if __name__ == '__main__':
    model = Net(2, 1)
    print(list(model._parameters))
    print(list(model._modules))
    for idx, m in enumerate(model.named_modules()):
        print(idx, '->', m)
    print(model(Tensor(np.ones(shape=(2,)))))