import numpy as np
from typing import List, NamedTuple, Callable, Optional, Union

Array_like = Union[float, list, np.ndarray]
def to_array(array_like:Array_like):
    if isinstance(array_like, np.ndarray):
        return array_like
    else:
        return np.array(array_like)

class Dependency(NamedTuple):
    tensor: 'Tensor'
    grad_fn: Callable[[np.ndarray], np.ndarray]

class Tensor:
    def __init__(self,
                data:Array_like,
                requires_grad:bool = False,
                depends_on:List[Dependency] = None,) -> None:
        self.data = to_array(data)
        self.requires_grad = requires_grad
        self.depends_on = depends_on or []
        self.shape = self.data.shape
        self.grad:Optional['Tensor'] = None

        if self.requires_grad:
            self.zero_grad()

    def __repr__(self):
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"

    def zero_grad(self) -> None:
        self.grad = Tensor(np.zeros_like(self.data))

    def backward(self, grad:'Tensor' = None):
        assert self.requires_grad, "somehow called backwards on tensor that doesn't require gradient"

        if grad is None:
            if self.shape == ():
                grad = Tensor(1.)
            else:
                raise RuntimeError('grad must a specified for a non-0-dim tensor')
        
        self.grad.data = self.grad.data + grad.data
    
        for dependency in self.depends_on:
            backward_grad = dependency.grad_fn(grad.data)
            dependency.tensor.backward(Tensor(backward_grad))

    def sum(self) -> None:
        return tensor_sum(self)

def tensor_sum(t: Tensor) -> Tensor:
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

def add(t1: Tensor, t2: Tensor) -> Tensor:
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

def multiply(t1:Tensor, t2:Tensor) -> Tensor:
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
        def grad_fn1(grad: np.ndarray) -> np.ndarray:
            grad = grad * t1.data
            ndim_added = grad.ndim - t2.data.ndim
            for _ in range(ndim_added):
                grad = grad.sum(axis=0)

            for i, dim in enumerate(t2.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)

            return grad
        
        depends_on.append(Dependency(t2, grad_fn1))

    return Tensor(data, requires_grad, depends_on)

def neg(t: Tensor) -> Tensor:
    data = -t.data
    requires_grad = t.requires_grad
    if t.requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return -grad
        depends_on = [Dependency(t, grad_fn)]
    else:
        depends_on = []
    return Tensor(data, requires_grad, depends_on)

def sub(t1: Tensor, t2:Tensor) -> Tensor:
    return add(t1, neg(t2))