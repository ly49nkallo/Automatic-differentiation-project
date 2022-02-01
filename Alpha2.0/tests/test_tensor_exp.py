import unittest
import pytest
import numpy as np

from autograd.tensor import Tensor
'''
class TestTensorExp(unittest.TestCase):
    def test_simple_exp(self):
        t1 = Tensor([1, 2, 3], requires_grad=True)
        t2 = t1.exp()
        s = t2.data
        t2.backward(Tensor([1,1,1]))

        np.testing.assert_array_equal(np.array([np.exp(1), np.exp(2), np.exp(3)]), t1.grad.data) 

    def test_softmax_with_grad(self):
        t1 = Tensor([1, 2, 3], requires_grad=True)
        t2 = t1.softmax()

        t2.backward(Tensor([1,1,1]))

        assert t1.grad.data.tolist() == [3, 3, 3]
'''