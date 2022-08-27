from re import T
import unittest
import pytest
import numpy as np

from autograd.tensor import Tensor
from autograd.functional import *
'''
class TestView(unittest.TestCase):
    def test_simple_view(self):
        t1 = Tensor(np.zeros([5,2]), requires_grad=True)
        t2 = t1.view(2,5)
        t2.backward(Tensor(np.ones_like(t2.data)))

        assert t1.grad.data.tolist() == np.ones([5,2]).tolist()

    def test_1D_view(self):
        t1 = Tensor(np.zeros(5), requires_grad=True)
        t2 = t1.view(1,-1)
        assert t2.shape == (1,5)

        gradient = Tensor([1,5,2,3,2])
        t2.backward(gradient.view(-1,1))

        #assert t1.grad.data.tolist() == gradient.data.tolist()

    def test_flatten(self):
        t1 = Tensor(np.random.randn(2,3,4,5), requires_grad=True)
        t2 = t1.view(-1)
        t3 = t2.sum()
        t3.backward()
        assert t2.size == 2*3*4*5
        assert t1.grad.data.tolist() == np.ones((2,3,4,5)).tolist()
        del t1, t2

        t1 = Tensor(np.random.randn(32,64), requires_grad=True)
        t2 = t1.view(-1)
        t3 = t2.sum()
        t3 = t3 * t3.view(*t3.shape)
        t3.backward()

        assert t1.grad.data.tolist() == np.ones_like(t1.data).tolist()\
'''

