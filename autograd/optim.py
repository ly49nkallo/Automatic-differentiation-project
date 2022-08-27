from ast import Param
from typing import Iterator
from autograd.parameter import Parameter
from autograd.tensor import Tensor
import numpy as np

"""
Optimizers go here
"""

class Optimizer_base:
    def __init__(self, parameters:Iterator[Parameter]):
        if isinstance(parameters, Tensor):
            raise TypeError("params argument given to the optimizer should be "
                            "an iterable of Tensors or dicts, but got " +
                            type(parameters))
        if hasattr(parameters, '__call__'):
            raise TypeError("params argument given to the optimizer should be "
                            "an iterable of Tensors or dicts, but got " +
                            type(parameters) + "\nDid you acidentally pass in the\
                            autograd.module.parameters function object instead of\
                            calling it?")
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
                print('set to none')
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

class Momentum(Optimizer_base):
    def __init__(self, params:Iterator[Parameter], lr: float = 0.01, gamma:float = 0.9) -> None:
        super().__init__(params)
        self.lr = lr
        self.gamma = gamma
        #must let each parameter store it's own v values

    def step(self) -> None:
        for parameter in self.parameters:
            v = parameter.grad * self.lr - self.gamma * parameter.v
            assert isinstance(v, Tensor)
            parameter -= v
            parameter.v = v

class Adam(Optimizer_base):
    def __init__(self, params:Iterator[Parameter], lr: float = 0.01):
        super().__init__(params)
        self.lr = lr
        self.b1 = 0.9
        self.b2 = 0.999
        self.eps = 1e-8
        self.timestep = 0
        
    def step(self) -> None:
        self.timestep += 1
        for parameter in self.parameters:
            g = parameter.grad.data
            m = self.b1 * parameter.m + (1-self.b1) * g
            v = self.b2 * parameter.v + (1-self.b2) * (g * g)
            mhat = m / (1 - self.b1 ** self.timestep)
            vhat = v / (1 - self.b2 ** self.timestep)
            parameter.v = v
            parameter.m = m
            #TODO python does not pass by reference, only by assignment. Altering the parameter
            #   here creates a new variable called "parameter" and gives it the assignment.
            #   However, inplace works because it calls a function to alter the data in the variable in place.
            #   This does not traditionally work with our system of backpropagation because we need to preserve every node
            #   exactly as it was. 
            #   When we make a GAN, we do not want to actually delete the gradient of the parameter when we alter it. 
            parameter -= (self.lr / (np.sqrt(vhat) + self.eps) * mhat)


