import numpy as np
from typing import Optional
from autograd.tensor import Tensor

class Parameter(Tensor):
    def __init__(self, *shape) -> None:
        data = np.random.randn(*shape)
        super().__init__(data, requires_grad=True)
        # stores most recent gradient descent step
        # defined for Momentum
        self.v:np.ndarray = np.zeros(self.shape)
        self.m:np.ndarray = np.zeros(self.shape)
        

    def zero_grad(self) -> None:
        super().zero_grad()
        