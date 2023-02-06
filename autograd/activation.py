from autograd.module import Module
from autograd.functional import tanh
class Sigmoid(Module):
    r'''Sigmoid Activation Function'''
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.logsig()

class Tanh(Module):
    r'''Tanh Activation Function'''
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return tanh(x)

class ReLU(Module):
    r'''Relu Activation Function'''
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return x.relu()

class Softmax(Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x):
        return x.softmax()
    
class Experimental(Module):
    r'''Experimental Activation Function'''
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x):
        return 