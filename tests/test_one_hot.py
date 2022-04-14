import unittest
import pytest
import numpy as np

from autograd.tensor import Tensor
from autograd.functional import one_hot_encode

class TestTensorOneHot(unittest.TestCase):
    def test_simple_onehot(self):
        t1 = Tensor([1,2,3], requires_grad=True)
        t1 = one_hot_encode(t1)

        assert t1.data.tolist() == np.array([[0, 1, 0, 0],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]]).astype(float).tolist()

        del t1

        t1 = Tensor(1., requires_grad = True)
        t2 = one_hot_encode(t1)
        t3 = one_hot_encode(t1, dtype=float, squeeze=False)

        assert t2.data.tolist() == [0, 1]
        assert t3.data.tolist() == [[0., 1.]]
        
        assert t2.grad is None
        assert t3.grad is None

        del t1, t2, t3

        t1 = Tensor([[[[[1,2,3,4,5]]]]], requires_grad = False)
        t2 = one_hot_encode(t1)
        t3 = one_hot_encode(t1, dtype=int, squeeze = False)
        
        assert t2.data.tolist() == one_hot_encode(Tensor([1,2,3,4,5])).data.tolist()
        assert t3.data.tolist() == t2.data.tolist()