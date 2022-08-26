import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
import time

from autograd.tensor import Tensor
from autograd.optim import SGD
from autograd.module import Module, Linear
from autograd.activation import Sigmoid, Tanh, Softmax
from autograd.functional import *

class testClass() :
    def __init__(self, data:list) :
        self.data = data

    def inc(self) :
        for d in self.data:
            d = d + 1
    
def main():
    c = testClass([1,2,3,4,5,6])
    c.inc()
    print(c.data)

main()