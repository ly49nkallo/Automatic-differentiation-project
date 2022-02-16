import unittest
import pytest
import numpy as np

from autograd.tensor import Tensor

class TestTensorSum(unittest.TestCase):
    def test_simple_sum(self):
        t1 = Tensor([1, 2, 3], requires_grad=True)
        t2 = t1.sum()

        t2.backward()

        assert t1.grad.data.tolist() == [1, 1, 1]

    def test_sum_with_grad(self):
        t1 = Tensor([1, 2, 3], requires_grad=True)
        t2 = t1.sum()

        t2.backward(Tensor(3))

        assert t1.grad.data.tolist() == [3, 3, 3]

    def test_sum_along_axis(self):
        t1 = Tensor(np.arange(3**4).reshape(3,3,3,3), requires_grad=True)
        t2 = t1.sum(axis = 1)

        t2.backward(Tensor([[[1,2,3] for _ in range(3)] for _ in range(3)]))

        #assert t2.data.tolist() == [9, 12, 15]
        print(t2.data)
        print(t1.grad.shape)
        
        assert t1.grad.data.tolist() == [[[[1,2,3] for _ in range(3)] for _ in range(3)] for _ in range(3)]