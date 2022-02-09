import unittest
import pytest
import numpy as np

from autograd.tensor import Tensor

class TestTensorLog(unittest.TestCase):
    def test_simple_log(self):
        t1 = Tensor([1,2,3], requires_grad=True)
        t2 = t1.log()
        
        assert t2.data.tolist() == [np.log(a+1) for a in range(3)]

        t2.backward(Tensor([1,1,1]))

        assert t1.grad.data.tolist() == [1., 1/2, 1/3]
