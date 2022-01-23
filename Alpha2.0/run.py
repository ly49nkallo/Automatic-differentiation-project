from module import Tanh, Linear, Module
from tensor import Parameter
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
    print(model._parameters)
    print(model._modules)
    for idx, m in enumerate(model.named_modules()):
        print(idx, '->', m)
    print(dir(model))