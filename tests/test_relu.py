import unittest
import numpy as np

from autograd.tensor import Tensor
import autograd.functional as F
from autograd.activation import *

class TestTensorReLU(unittest.TestCase):
    def test_maximum(self):
        a = np.arange(20).reshape(5,4) - 10
        b = np.maximum(a, 0)
        assert b.shape == (5,4) and b.max() == 10-1 and b.min() == 0
        del a, b
        
    def test_simple_ReLU(self):
        t1 = Tensor(np.arange(25).reshape(5,5) - 10, requires_grad = True)
        assert t1.max == 14 and t1.min == -10
        rlu = ReLU()
        t2 = rlu(t1)
        assert t2.max == 14 and t2.min == 0
        t3 = t2.sum()
        t3.backward()
        print(t2)
        assert False