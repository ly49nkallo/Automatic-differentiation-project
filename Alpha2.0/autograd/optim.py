from ast import Param
from typing import Iterator
from autograd.parameter import Parameter
from autograd.tensor import Tensor

"""
Optimizers go here
"""

class Optimizer_base:
    def __init__(self, parameters:Iterator[Parameter]):
        if isinstance(parameters, Tensor):
            raise TypeError("params argument given to the optimizer should be "
                            "an iterable of Tensors or dicts, but got " +
                            type(parameters))

        self.parameters = list(parameters)
        if len(self.parameters) == 0:
            raise ValueError("optimizer recieved no parameters")

    def step(self) -> None:
        '''Optimizer subclasses must implement the step method'''
        raise NotImplementedError("Optimizer subclasses must implement the step method")

    def zero_grad(self, set_to_none:bool = False) -> None:
        '''Zero out every parameter's gradient that is stored in this optimizer'''
        for parameter in self.parameters:
            if set_to_none:
                parameter.grad = None
            else:
                parameter.zero_grad()

class SGD(Optimizer_base):
    def __init__(self, params:Iterator[Parameter], lr: float = 0.01) -> None:
        super().__init__(params)
        self.lr = lr

    def step(self) -> None:
        for parameter in self.parameters:
            parameter -= parameter.grad * self.lr