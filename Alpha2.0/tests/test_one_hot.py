import unittest
import pytest
import numpy as np

from autograd.tensor import Tensor
from autograd.functional import one_hot_encode

class TestTensorLog(unittest.TestCase):
    def test_simple_log(self):
        t1 = Tensor([1,2,3], requires_grad=True)
        t1 = one_hot_encode(t1)

        assert t1.data.tolist() == np.array([[0, 1, 0, 0],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]]).astype(float).tolist()
