import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
import time

from autograd.tensor import Tensor
from autograd.optim import SGD
from autograd.module import Module, Linear
from autograd.activation import Sigmoid, Tanh, Softmax
from autograd.functional import *

a = Tensor(np.ones((5,5)), requires_grad=True)
b = a.view(25)
b.backward(np.zeros((25,)))
print(a.grad, b.grad)