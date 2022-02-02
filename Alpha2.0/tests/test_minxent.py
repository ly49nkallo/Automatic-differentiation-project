import unittest
import pytest
import numpy as np

from autograd.tensor import Tensor
from autograd.functional import minxent, binxent, mse

class TestTensorMinxent(unittest.TestCase):
    def test_simple_minxent(self):
        t1 = Tensor([1,2,3], requires_grad=True)
        t2 = t1.softmax()

        assert np.abs(np.sum(t2.data) - 1.) < 1e-3 # there is still some numerical instability
        labels = Tensor(np.ones(3))
        t3 = minxent(t2, labels)

        print(t3.data)
        print(t2.grad.data)
        assert False
        