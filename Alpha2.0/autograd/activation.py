from autograd.module import Module

class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.tansig()

class Tanh(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.tanh()

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