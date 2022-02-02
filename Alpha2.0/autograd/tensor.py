from warnings import WarningMessage, warn
import numpy as np
from typing import List, NamedTuple, Callable, Optional, Union

Array_like = Union[float, list, np.ndarray]
Tensorable = Union['Tensor', float, np.ndarray]

def ensure_array(array_like:Array_like):
    if isinstance(array_like, np.ndarray):
        return array_like
    else:
        return np.array(array_like)

def ensure_tensor(tensorable:Tensorable) -> 'Tensor':
    if isinstance(tensorable, Tensor):
        return tensorable
    else:
        return Tensor(tensorable)

class Dependency(NamedTuple):
    tensor: 'Tensor'
    grad_fn: Callable[[np.ndarray], np.ndarray]

class Tensor:
    def __init__(self,
                data:Array_like,
                requires_grad:bool = False,
                depends_on:List[Dependency] = None,) -> None:
        self.data = ensure_array(data)
        self.requires_grad = requires_grad
        self.depends_on = depends_on or []
        self.shape = self.data.shape
        self.grad:Optional['Tensor'] = None

        if self.requires_grad:
            self.zero_grad()

    @property
    def data(self) -> np.ndarray:
        return self._data

    @data.setter
    def data(self, value):
        self._data = value
        # setting data invalidates the tensor gradient
        #warn('Tensors are normally immutable - setting value invalidates it')
        self.grad = None

    def __repr__(self):
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"

    def truth(self):
        warn('truth value of tensor defaults to its data')
        return self.data.truth()

    def __add__(self, other:Tensorable) -> 'Tensor':
        return _add(self, ensure_tensor(other))

    def __radd__(self, other:Tensorable) -> 'Tensor':
        return _add(ensure_tensor(other), self)

    def __iadd__(self, other:Tensorable) -> 'Tensor':
        self.data = self.data + ensure_tensor(other).data
        return self

    def __neg__(self) -> 'Tensor':
        return _neg(self)

    def __mul__(self, other:Tensorable) -> 'Tensor':
        return _multiply(self, ensure_tensor(other))

    def __rmul__(self, other:Tensorable) -> 'Tensor':
        return _multiply(ensure_tensor(other), self)

    def __imul__(self, other:Tensorable) -> 'Tensor':
        self.data = self.data * ensure_tensor(other).data
        return self

    def __matmul__(self, other) -> 'Tensor':
        return _matmul(self, other)

    def __sub__(self, other:Tensorable) -> 'Tensor':
        return _sub(self, ensure_tensor(other))

    def __rsub__(self, other:Tensorable) -> 'Tensor':
        return _sub(ensure_tensor(other), self)

    def __isub__(self, other:Tensorable) -> 'Tensor':
        self.data = self.data - ensure_tensor(other).data
        # invalidate gradient
        self.grad = None
        return self

    def __truediv__(self, other:Tensorable) -> 'Tensor':
        return _truediv(self, ensure_tensor(other))
    
    def __pow__(self, other:Tensorable) -> 'Tensor':
        if int(ensure_tensor(other).data) == 2:
            return self * self
        else:
            raise NotImplementedError("High degree exponentials not implemented yet!")
    def __getitem__(self, idxs) -> 'Tensor':
        return _slice(self, idxs)

    def zero_grad(self) -> None:
        self.grad = Tensor(np.zeros_like(self.data))

    def backward(self, grad:'Tensor' = None):
        assert self.requires_grad, "called backwards on tensor that doesn't require gradient"

        if grad is None:
            if self.shape == ():
                grad = Tensor(1.)
            else:
                raise RuntimeError('grad must a specified for a non-0-dim tensor')
        
        self.grad.data = self.grad.data + grad.data #type: ignore
    
        for dependency in self.depends_on:
            backward_grad = dependency.grad_fn(grad.data)
            dependency.tensor.backward(Tensor(backward_grad))

    def sum(self) -> 'Tensor':
        return _tensor_sum(self)

    def exp(self) -> 'Tensor':
        return _exp(self)

    def tansig(self) -> 'Tensor':
        return _tansig(self)

    def tanh(self) -> 'Tensor':
        return _tanh(self)

    def relu(self) -> 'Tensor':
        return _relu(self)
    
    def identity(self) -> 'Tensor':
        return _identity(self)

    def softmax(self) -> 'Tensor':
        return _softmax(self)

'''TENSOR FUNCTIONS'''

def _tensor_sum(t: Tensor) -> Tensor:
    "Wraps the np.sum and returns a zero-tensor"
    data = t.data.sum()
    requires_grad = t.requires_grad

    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            '''
                grad is a zero-tensor so each element contributes that much
            '''
            return grad * np.ones_like(t.data)

        depends_on = [Dependency(t, grad_fn)]
    else:
        depends_on = []
    
    return Tensor(data, requires_grad, depends_on)

def _add(t1: Tensor, t2: Tensor) -> Tensor:
    # f(x, y) = x+y
    # Dxf(x,y) = 1
    data = t1.data + t2.data
    # if either component requires gradient computation, the sum must require it too
    requires_grad = t1.requires_grad or t2.requires_grad
    depends_on:List[Dependency] = []

    if t1.requires_grad:
        def grad_fn1(grad: np.ndarray) -> np.ndarray:
            #sum out added dims
            ndim_added = grad.ndim - t1.data.ndim
            for _ in range(ndim_added):
                grad = grad.sum(axis=0)
            
            # sum across broadcasted (but non-added-dims)
            for i, dim in enumerate(t1.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)
            
            return grad

        depends_on.append(Dependency(t1, grad_fn1))

    if t2.requires_grad:
        def grad_fn2(grad: np.ndarray) -> np.ndarray:
            #sum out added dims
            ndim_added = grad.ndim - t2.data.ndim
            for _ in range(ndim_added): 
                grad = grad.sum(axis=0)
            
            # sum across broadcasted (but non-added-dims)
            for i, dim in enumerate(t2.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)
            
            return grad

        depends_on.append(Dependency(t2, grad_fn2))

    return Tensor(data, requires_grad, depends_on)

def _multiply(t1:Tensor, t2:Tensor) -> Tensor:
    # f(x,y) = xy
    # Dxf(x,y) = cx
    # Dyf(x,y) = cy
    data = t1.data * t2.data
    requires_grad = t1.requires_grad or t2.requires_grad
    depends_on: List[Dependency] = []

    if t1.requires_grad:
        def grad_fn1(grad: np.ndarray) -> np.ndarray:
            grad = grad * t2.data
            ndim_added = grad.ndim - t1.data.ndim
            for _ in range(ndim_added):
                grad = grad.sum(axis=0)

            for i, dim in enumerate(t1.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)

            return grad
        
        depends_on.append(Dependency(t1, grad_fn1))

    if t2.requires_grad:
        def grad_fn2(grad: np.ndarray) -> np.ndarray:
            grad = grad * t1.data
            ndim_added = grad.ndim - t2.data.ndim
            for _ in range(ndim_added):
                grad = grad.sum(axis=0)

            for i, dim in enumerate(t2.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)

            return grad
        
        depends_on.append(Dependency(t2, grad_fn2))

    return Tensor(data, requires_grad, depends_on)

def _neg(t: Tensor) -> Tensor:
    data = -t.data
    requires_grad = t.requires_grad
    if t.requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return -grad
        depends_on = [Dependency(t, grad_fn)]
    else:
        depends_on = []
    return Tensor(data, requires_grad, depends_on)

def _sub(t1: Tensor, t2:Tensor) -> Tensor:
    return t1 + -t2

def _matmul(t1: Tensor, t2: Tensor) -> Tensor:
    """
    if t1 is (n1, m1) and t2 is (m1, m2), then t1 @ t2 is (n1, m2)
    so grad3 is (n1, m2)
    if t3 = t1 @ t2, and grad3 is the gradient of some function wrt t3, then
        grad1 = grad3 @ t2.T
        grad2 = t1.T @ grad3
    """
    #if t1.data.ndim == 1:

    data = t1.data @ t2.data
    requires_grad = t1.requires_grad or t2.requires_grad

    depends_on: List[Dependency] = []

    if t1.requires_grad:
        def grad_fn1(grad: np.ndarray) -> np.ndarray:
            return grad @ t2.data.T

        depends_on.append(Dependency(t1, grad_fn1))

    if t2.requires_grad:
        def grad_fn2(grad: np.ndarray) -> np.ndarray:
            return t1.data.T @ grad
        depends_on.append(Dependency(t2, grad_fn2))

    return Tensor(data,
                  requires_grad,
                  depends_on)

def _truediv(t1:Tensor, t2:Tensor) -> Tensor:
    # f(x,y) = x/y
    # Dxf(x,y) = Dx x/c = 1/c * Dxx = 1/c
    # Dyf(x,y) = Dy c/y = c * Dy 1/y = c * (-x/y**2)

    # https://www.wolframalpha.com/input?i=partial+derivative+of+x%2Fy

    data = t1.data / t2.data
    requires_grad = t1.requires_grad or t2.requires_grad
    depends_on: List[Dependency] = []

    if t1.requires_grad:
        def grad_fn1(grad: np.ndarray) -> np.ndarray:
            # D x/c 
            grad = grad / t2.data
            ndim_added = grad.ndim - t1.data.ndim
            for _ in range(ndim_added):
                grad = grad.sum(axis=0)

            for i, dim in enumerate(t1.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)

            return grad
        
        depends_on.append(Dependency(t1, grad_fn1))

    if t2.requires_grad:
        def grad_fn2(grad: np.ndarray) -> np.ndarray:
            grad = grad * -(t1.data / (t2.data ** 2))
            ndim_added = grad.ndim - t2.data.ndim
            for _ in range(ndim_added):
                grad = grad.sum(axis=0)

            for i, dim in enumerate(t2.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)

            return grad
        
        depends_on.append(Dependency(t2, grad_fn2))

    return Tensor(data, requires_grad, depends_on)

def _slice(t: Tensor, idxs) -> Tensor:
    data = t.data[idxs]
    requires_grad = t.requires_grad

    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            bigger_grad = np.zeros_like(data)
            bigger_grad[idxs] = grad
            return bigger_grad

        depends_on = Dependency(t, grad_fn)
    else:
        depends_on = []

    return Tensor(data, requires_grad, depends_on)

def _tansig(t:Tensor) -> Tensor:
    data = 1 / (1 + np.exp(-t.data))
    requires_grad = t.requires_grad
    if requires_grad:
        def grad_fn(grad:np.ndarray) -> np.ndarray:
            return grad * (data * (1 - data))
        
        depends_on = [Dependency(t, grad_fn)]
    else:
        depends_on = []

    return Tensor(data, requires_grad, depends_on)

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

def _exp(t:Tensor) -> Tensor:
    # d(e**x)/dx = 
    data = np.exp(t.data)
    requires_grad = t.requires_grad
    if requires_grad:
        # the derivative of the natural exponential function is itself
        depends_on = [Dependency(t, lambda grad: grad * data)]
    else:
        depends_on = []
    return Tensor(data, requires_grad, depends_on)

def _softmax(t:Tensor) -> Tensor:
    return _exp(t) / (_exp(t).sum())

def _mse(t:Tensor) -> Tensor:
    pass
