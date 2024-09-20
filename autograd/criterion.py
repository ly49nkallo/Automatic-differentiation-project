from autograd.tensor import Tensor
from autograd.module import Module
from autograd.functional import *

class MSE_loss(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        raise NotImplementedError

class CrossEntropy_Loss(Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self , x):
        raise NotImplementedError
    
class L1_Loss(Module):
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, x):
        return L1Loss(x)
    
class MinXEnt_Loss(Module):
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, x:Tensor, y:Tensor):
        return minxent(x, y)
    
class BinXEnt_Loss(Module):
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, x:Tensor, y:Tensor):
        return BCELoss(x, y)