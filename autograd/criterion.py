from autograd.tensor import Tensor
from autograd.module import Module
from autograd.functional import *

class MSE_loss(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        raise NotImplementedError
        pass

class CrossEntropyLoss(Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self , x):
        raise NotImplementedError
        pass