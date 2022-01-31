from autograd.tensor import Tensor, Dependency
import numpy as np

def _tanh(t:Tensor) -> Tensor:
    data = (np.exp(t.data) - np.exp(-t.data) / np.exp(t.data) + np.exp(-t.data))
    requires_grad = t.requires_grad
    if requires_grad:
        def grad_fn(grad:np.ndarray) -> np.ndarray:
            return grad * (1- (data * data))
        depends_on = [Dependency(t, grad_fn)]
    else:
        depends_on = []
    
    return Tensor(data, requires_grad, depends_on)

def _relu(t:Tensor) -> Tensor:
    data = np.maximum(t.data, np.zeros_like(t.data))
    requires_grad = t.requires_grad
    if requires_grad:
        def grad_fn(grad:np.ndarray) -> np.ndarray:
            # the derivative of relu is 0 if x<0 and 1 if x>0
            return np.maximum(grad, np.zeros_like(grad))
        depends_on = [Dependency(t, grad_fn)]
    else:
        depends_on = []

    return Tensor(data, requires_grad, depends_on)

def _identity(t:Tensor) -> Tensor:
    return Tensor(t.data, t.requires_grad, [Dependency(t, lambda x: x)])

def _softmax(t:Tensor) -> Tensor:
    data = np.exp(t.data) / (np.sum(np.exp(t.data)))

def _mse(t:Tensor) -> Tensor: pass