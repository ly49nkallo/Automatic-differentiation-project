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

class Node(NamedTuple):
    tensor: 'Tensor'
    grad_fn: Callable[[np.ndarray], np.ndarray]

class Tensor:
    def __init__(self,
                data:Array_like,
                requires_grad:bool = False,
                parent_nodes:List[Node] = None,) -> None:
        self.data = ensure_array(data)
        self.requires_grad = requires_grad
        self.parent_nodes = parent_nodes or []
        self.shape = self.data.shape
        self.grad:Optional['Tensor'] = None

        if self.requires_grad:
            self.zero_grad()

    @property
    def data(self) -> np.ndarray:
        return self._data

    @property
    def size(self) -> int:
        return self.data.size

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
    
        for parent in self.parent_nodes:
            backward_grad = parent.grad_fn(grad.data)
            parent.tensor.backward(Tensor(backward_grad))

    '''Tensor operations'''

    def sum(self, axis:Optional[int] = None) -> 'Tensor':
        return _tensor_sum(self, axis=axis)

    def exp(self) -> 'Tensor':
        return _exp(self)

    def logsig(self) -> 'Tensor':
        return _logsig(self)

    def tanh(self) -> 'Tensor':
        return _tanh(self)

    def relu(self) -> 'Tensor':
        return _relu(self)
    
    def identity(self) -> 'Tensor':
        return _identity(self)

    def log(self) -> 'Tensor':
        return _log(self)

    def abs(self) -> 'Tensor':
        return _abs(self)

    def transpose(self) -> 'Tensor':
        return _transpose(self)

'''TENSOR FUNCTIONS'''

def _tensor_sum(t: Tensor, axis:Optional[int] = None, keep_dims:bool = False) -> Tensor:
    "Wraps the np.sum and returns a zero-tensor"
    data = t.data.sum(axis=axis,) #keepdims=keep_dims)
    requires_grad = t.requires_grad

    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            if axis is None:
                '''+
                    grad is a zero-tensor so each element contributes that much
                '''
                return grad * np.ones_like(t.data)
            else:
                ''' grad is a tensor that is the same shape as t.data minus the summed out axis'''
                print('shape', t.shape)
                shape = t.shape[:axis]+t.shape[axis+1:]
                print('shape', shape)
                return grad * np.ones(shape)
        parent_nodes = [Node(t, grad_fn)]
    else:
        parent_nodes = []
    
    return Tensor(data, requires_grad, parent_nodes)
    

def _add(t1: Tensor, t2: Tensor) -> Tensor:
    # f(x, y) = x+y
    # Dxf(x,y) = 1
    data = t1.data + t2.data
    # if either component requires gradient computation, the sum must require it too
    requires_grad = t1.requires_grad or t2.requires_grad
    parent_nodes:List[Node] = []

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

        parent_nodes.append(Node(t1, grad_fn1))

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

        parent_nodes.append(Node(t2, grad_fn2))

    return Tensor(data, requires_grad, parent_nodes)

def _multiply(t1:Tensor, t2:Tensor) -> Tensor:
    # f(x,y) = xy
    # Dxf(x,y) = cx
    # Dyf(x,y) = cy
    data = t1.data * t2.data
    requires_grad = t1.requires_grad or t2.requires_grad
    parent_nodes: List[Node] = []

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
        
        parent_nodes.append(Node(t1, grad_fn1))

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
        
        parent_nodes.append(Node(t2, grad_fn2))

    return Tensor(data, requires_grad, parent_nodes)

def _neg(t: Tensor) -> Tensor:
    data = -t.data
    requires_grad = t.requires_grad
    if t.requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return -grad
        parent_nodes = [Node(t, grad_fn)]
    else:
        parent_nodes = []
    return Tensor(data, requires_grad, parent_nodes)

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

    parent_nodes: List[Node] = []

    if t1.requires_grad:
        def grad_fn1(grad: np.ndarray) -> np.ndarray:
            return grad @ t2.data.T

        parent_nodes.append(Node(t1, grad_fn1))

    if t2.requires_grad:
        def grad_fn2(grad: np.ndarray) -> np.ndarray:
            return t1.data.T @ grad
        parent_nodes.append(Node(t2, grad_fn2))

    return Tensor(data,
                  requires_grad,
                  parent_nodes)

def _truediv(t1:Tensor, t2:Tensor) -> Tensor:
    # f(x,y) = x/y
    # Dxf(x,y) = Dx x/c = 1/c * Dxx = 1/c
    # Dyf(x,y) = Dy c/y = c * Dy 1/y = c * (-x/y**2)

    # https://www.wolframalpha.com/input?i=partial+derivative+of+x%2Fy

    data = t1.data / t2.data
    requires_grad = t1.requires_grad or t2.requires_grad
    parent_nodes: List[Node] = []

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
        
        parent_nodes.append(Node(t1, grad_fn1))

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
        
        parent_nodes.append(Node(t2, grad_fn2))

    return Tensor(data, requires_grad, parent_nodes)

def _slice(t: Tensor, idxs) -> Tensor:
    data = t.data[idxs]
    requires_grad = t.requires_grad

    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            bigger_grad = np.zeros_like(data)
            bigger_grad[idxs] = grad
            return bigger_grad

        parent_nodes = Node(t, grad_fn)
    else:
        parent_nodes = []

    return Tensor(data, requires_grad, parent_nodes)

def _logsig(t:Tensor) -> Tensor:
    try:
        data = 1 / (1 + np.exp(-t.data))
    except RuntimeWarning:
        print(t.data)
    requires_grad = t.requires_grad
    if requires_grad:
        def grad_fn(grad:np.ndarray) -> np.ndarray:
            return grad * (data * (1 - data))
        
        parent_nodes = [Node(t, grad_fn)]
    else:
        parent_nodes = []

    return Tensor(data, requires_grad, parent_nodes)

def _tanh(t:Tensor) -> Tensor:
    data = (np.exp(t.data) - np.exp(-t.data) / np.exp(t.data) + np.exp(-t.data))
    requires_grad = t.requires_grad
    if requires_grad:
        def grad_fn(grad:np.ndarray) -> np.ndarray:
            return grad * (1- (data * data))
        parent_nodes = [Node(t, grad_fn)]
    else:
        parent_nodes = []
    
    return Tensor(data, requires_grad, parent_nodes)

def _relu(t:Tensor) -> Tensor:
    data = np.maximum(t.data, np.zeros_like(t.data))
    requires_grad = t.requires_grad
    if requires_grad:
        def grad_fn(grad:np.ndarray) -> np.ndarray:
            # the derivative of relu is 0 if x<0 and 1 if x>0
            return grad * (t.data > 0)
        parent_nodes = [Node(t, grad_fn)]
    else:
        parent_nodes = []

    return Tensor(data, requires_grad, parent_nodes)

def _identity(t:Tensor) -> Tensor:
    return Tensor(t.data, t.requires_grad, [Node(t, lambda x: x)])

def _exp(t:Tensor) -> Tensor:
    # d(e**x)/dx = 
    data = np.exp(t.data)
    requires_grad = t.requires_grad
    if requires_grad:
        # the derivative of the natural exponential function is itself
        parent_nodes = [Node(t, lambda grad: grad * data)]
    else:
        parent_nodes = []
    return Tensor(data, requires_grad, parent_nodes)

number = Union[float, int, np.float32, np.float64, np.intp]

def _log(t:Tensor, base:Optional[number] = None):
    assert base is None, "non natural logorithims not implemented"
    data = np.log(t.data)
    requires_grad = t.requires_grad
    if requires_grad:
        parent_nodes = [Node(t, lambda grad: grad / t.data)]
    else:
        parent_nodes = []

    return Tensor(data, requires_grad, parent_nodes)

def _abs(t:Tensor) -> Tensor:
    data = np.abs(t.data)
    requires_grad = t.requires_grad
    if requires_grad:
        parent_nodes = [Node(t, 
                                lambda grad : grad * np.vectorize(lambda x: 2. * int(x >= 0) - 1.) (t.data))]
    else:
        parent_nodes = []

    return Tensor(data, requires_grad, parent_nodes)

#@TODO

def _expand_dim(t:Tensor, dim:int):
    raise NotImplementedError

def _transpose(t:Tensor):
    raise NotImplementedError

def _view(t:Tensor):
    raise NotImplementedError
