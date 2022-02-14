from autograd.module import Module
from autograd.functional import tanh
class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.logsig()

class Tanh(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return tanh(x)

class ReLU(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return x.relu()

class Softmax(Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x):
        return x.softmax()