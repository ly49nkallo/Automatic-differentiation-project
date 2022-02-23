import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
import time

from autograd.tensor import Tensor
from autograd.optim import SGD
from autograd.module import Module, Linear
from autograd.activation import Sigmoid, Tanh, Softmax
from autograd.functional import *

plt.plot(np.arange(100)**2)
plt.plot(np.arange(100)**2 // 2)
plt.title("test title")
plt.legend(('blue', 'orange'), title='title')
plt.show()
