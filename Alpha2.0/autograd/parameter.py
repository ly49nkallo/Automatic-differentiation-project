import numpy as np
from typing import Optional
from autograd.tensor import Tensor

class Parameter(Tensor):
    def __init__(self, *shape, v:Optional[np.ndarray] = None) -> None:
        data = np.random.randn(*shape)
        super().__init__(data, requires_grad=True)
        # stores most recent gradient descent step
        # defined for Momentum
        if v is None:
            v = Tensor(np.zeros(shape))
        elif isinstance(v, Tensor): self.v = v
        else: self.v = Tensor(v)

    def set_v(self, v:np.ndarray) -> None:
        if isinstance(v, Tensor):
            self.v = v
        else:
            self.v = Tensor(v)

    def zero_grad(self) -> None:
        super().zero_grad()
        self.v = Tensor(np.zeros(self.shape))