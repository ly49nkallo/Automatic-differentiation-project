import numpy as np
from typing import Optional
from autograd.tensor import Tensor

class Parameter(Tensor):
    def __init__(self, *shape, v:Tensor = None) -> None:
        data = np.random.randn(*shape)
        super().__init__(data, requires_grad=True)
        # stores most recent gradient descent step
        # defined for Momentum
        if v is None:
            v = Tensor(np.zeros(shape))
        self.v = v

    def zero_grad(self) -> None:
        super().zero_grad()
        self.v = Tensor(np.zeros(self.shape))