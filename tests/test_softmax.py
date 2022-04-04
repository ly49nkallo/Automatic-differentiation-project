import unittest
import pytest
import numpy as np

from autograd.tensor import Tensor
import autograd.functional as F

class TestTensorSoftmax(unittest.TestCase):
    def test_simple_softmax(self):
        t1 = Tensor(np.ones((1, 14)), requires_grad=True)
        t2 = F.stable_softmax(t1.data)
        
        assert t2.data.tolist() == [[1/14] * 14]