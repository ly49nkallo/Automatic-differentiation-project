from .module import Tanh, Linear, Module

class Net(Module):
    def __init__(self, in_features, out_features):
        self.linear = Linear(in_features, out_features)
        self.act = Tanh()

    def forward(self, x):
        x = self.linear(x)
        x = self.act(x)
        return x

if __name__ == '__main__':
    model = Net(2, 1)
        