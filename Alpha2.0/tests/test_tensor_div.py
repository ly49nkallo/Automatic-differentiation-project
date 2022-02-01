import unittest
import pytest

import numpy as np

from autograd.tensor import Tensor

class TestTensorDiv(unittest.TestCase):
    def test_simple_div(self):
        t1 = Tensor([1, 2, 3], requires_grad=True)
        t2 = Tensor([4, 5, 6], requires_grad=True)

        t3 = t1 * t2

        assert t3.data.tolist() == [4, 10, 18]

        t3.backward(Tensor([-1., -2., -3.]))

        assert t1.grad.data.tolist() == [-4, -10, -18]
        assert t2.grad.data.tolist() == [-1,  -4,  -9]

        #t1 = t1 / 10
        #assert t1.grad is None

        #np.testing.assert_array_almost_equal(t1.data, [0.1, 0.2, 0.3])

        t4 = Tensor([1, 2, 3], requires_grad=True)
        t5 = Tensor([0.25, 0.4, 0.5], requires_grad=True)

        t6 = t4 / t5

        assert t6.data.tolist() == [4, 5, 6]

        t6.backward(Tensor([-1., -2., -3.]))

        assert t4.grad.data.tolist() == [-4, -10, -18]
        assert t5.grad.data.tolist() == [-1,  -4,  -9]

        t1 *= 0.1
        assert t1.grad is None

        np.testing.assert_array_almost_equal(t1.data, [0.1, 0.2, 0.3])

    def test_broadcast_div(self):
        t1 = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad = True)  # (2, 3)
        t2 = Tensor([7, 8, 9], requires_grad = True)               # (3,)

        t3 = t1 * t2   # shape (2, 3)

        assert t3.data.tolist() == [[7, 16, 27], [28, 40, 54]]

        t3.backward(Tensor([[1, 1, 1], [1, 1, 1]]))

        assert t1.grad.data.tolist() == [[7, 8, 9], [7, 8, 9]]
        assert t2.grad.data.tolist() == [5, 7, 9]

    def test_broadcast_div2(self):
        t1 = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad = True)    # (2, 3)
        t2 = Tensor([[7, 8, 9]], requires_grad = True)               # (1, 3)

        t3 = t1 * t2

        assert t3.data.tolist() == [[7, 16, 27], [28, 40, 54]]

        t3.backward(Tensor([[1, 1, 1], [1, 1, 1]]))

        assert t1.grad.data.tolist() == [[7, 8, 9], [7, 8, 9]]
        assert t2.grad.data.tolist() == [[5, 7, 9]]