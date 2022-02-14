import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
import time

from autograd.tensor import Tensor
from autograd.optim import SGD
from autograd.module import Module, Linear
from autograd.activation import Sigmoid, Tanh, Softmax
from autograd.functional import *

t1 = Tensor([[0,0,0,0],
             [0,0,0,0]], requires_grad=True)
t2 = Tensor([1, 1])

t3 = nll(t1, t2)
print(t3)
t3.backward()
print(t1.grad.data)

