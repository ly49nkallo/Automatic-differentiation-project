from re import T
import unittest
import pytest
import numpy as np

from autograd.tensor import Tensor
from autograd.functional import *

class TestMinxent(unittest.TestCase):
    def test_simple_softmax(self):
        t1 = Tensor(np.ones((1,5)), requires_grad=True)
        t3 = softmax(t1, dim=1)

        print(np.ones((1,5))/ 5)
        assert t3.data.tolist() == (np.ones((1,5)) / 5) .tolist()

        t3.backward(Tensor([[1,1,1,1,1]]))
        print(t3.grad.data)
        print(t1.grad.data)
        assert t3.grad.data.tolist() == np.ones((1,5)).tolist()
        assert t1.grad.data.tolist() == np.zeros((1,5)).tolist()

        del t1, t3

        t1 = Tensor([[0, 100, 0, 0]], requires_grad=True)
        assert t1.shape == (1,4)
        t2 = softmax(t1, dim=1)

        np.testing.assert_array_almost_equal(t2.data, np.array([[0., 1., 0., 0.]]))

    def test_simple_minxent(self):
        t1 = Tensor([[0, 50, 0, 0]], requires_grad=True)
        t2 = Tensor([[1]])
        t3 = minxent(t1, t2)

        assert abs(float(t3.data) - 0.) < 0.001, t3

        t3.backward()

        print(t1.grad.data)
        print(t3.grad.data)

        #assert False

