import unittest
import pytest
import numpy as np

from autograd.tensor import Tensor

class TestTensorabs(unittest.TestCase):
    def test_simple_abs(self):
        t1 = Tensor([1,2,-3, -4, 5], requires_grad=True)
        t2 = t1.abs()
        
        assert t2.data.tolist() == [a+1 for a in range(5)]

        t2.backward(Tensor(np.ones(5)))

        assert t1.grad.data.tolist() == [1, 1, -1, -1, 1]

        del t1, t2

        t1 = Tensor([1,2,-3, -4, 5], requires_grad=True)
        t2 = t1.abs()

        t2.backward(Tensor([2,2,-2,1,-1]))

        assert t1.grad.data.tolist() == (np.array([1, 1, -1, -1, 1]) * np.array([2,2,-2,1,-1])).tolist()