from autograd.tensor import Tensor, Node, _log
#import autograd
#Tensor = autograd.tensor.Tensor
import numpy as np
from typing import Optional, Union

def tanh(t:Tensor) -> Tensor:
    #bugfix 14.3
    data = (np.exp(t.data) - np.exp(-t.data)) / (np.exp(t.data) + np.exp(-t.data))
    requires_grad = t.requires_grad
    if requires_grad:
        def grad_fn(grad:np.ndarray) -> np.ndarray:
            return grad * (1- (data * data))
        parent_nodes = [Node(t, grad_fn)]
    else:
        parent_nodes = []
    
    return Tensor(data, requires_grad, parent_nodes)

def logsig(t:Tensor) -> Tensor:
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

def relu(t:Tensor) -> Tensor:
    data = np.maximum(t.data, np.zeros_like(t.data))
    requires_grad = t.requires_grad
    if requires_grad:
        def grad_fn(grad:np.ndarray) -> np.ndarray:
            # the derivative of relu is 0 if x<0 and 1 if x>0
            return np.maximum(grad, np.zeros_like(grad))
        parent_nodes = [Node(t, grad_fn)]
    else:
        parent_nodes = []

    return Tensor(data, requires_grad, parent_nodes)

def identity(t:Tensor) -> Tensor:
    return Tensor(t.data, t.requires_grad, [Node(t, lambda x: x)])

def exp(t:Tensor) -> Tensor:
    return t.exp()

def softmax(t:Tensor, dim=1) -> Tensor:
    r''' expects t.shape to be (batch_size, classes)
        will default softmax along axis index 1'''
    # t.shape: (batch_size, num_classes)
    #TODO does not work with batch_size > 1
    return exp(t) / (exp(t)).sum(axis=dim)

def mse(predicted:Tensor, actual:Tensor, is_one_hot = True) -> Tensor: 
    if not is_one_hot:
        actual = one_hot_encode(actual, 10)
    errors = predicted - actual
    loss = (errors * errors).sum()
    return loss


# also called cross entropy loss due to it's usage by statistical analysis (minxent)
# https://gombru.github.io/assets/cross_entropy_loss/intro.png
def minxent(input:Tensor, target:Tensor, is_one_hot = False, dim=1) -> Tensor:
    r'''AKA Categorical Cross entropy loss by statastitians or negative log likelihood (NLL)
        Args:
            input (Tensor): the input tensor (preferably NOT softmaxed)
            target (Tensor): a tensor containing the ground truth (preferably one-hot vector)'''
    
    # input.shape (batch_size, num_classes)
    # target.shape (batch_size,)

    m = target.shape[0]
    if not is_one_hot:
        truth = one_hot_encode(target, num_of_classes=input.data.shape[-1])
    input = softmax(input)
    return -(log(input) * truth).sum()

def nll(input:Tensor, target:Tensor, dim=1) -> Tensor:
    r'''Negative-log-likelihood loss function with softmax
            Args:
                input (Tensor): a direct output of the network without softmax
                target (Tensor): a tensor containing labels
                
            Note: target shape's first dimension must be dimension num_classes'''
    # https://deepnotes.io/softmax-crossentropy
    # input.shape == (batch_size, num_classes)
    # ###labels.shape == (batch_size, 1)
    m = target.shape[0]
    p = stable_softmax(input.data)

    log_likelihood = -np.log(p[range(m), target.data])
    data = np.sum(log_likelihood) / m
    requires_grad = input.requires_grad
    if requires_grad:
        def grad_fn(grad:np.ndarray) -> np.ndarray:
            g = p.copy()
            g[range(m), target.data] -= 1
            g = g/m
            return grad * g
        parent_nodes = [Node(input, grad_fn)]
    else:
        parent_nodes = []
    return Tensor(data, requires_grad, parent_nodes) 

    
def stable_softmax(X:np.ndarray):
    if X.ndim < 2:
        raise ValueError(f"X must be shaped like (batch, *data), instead got tensor of shape {X.shape}")
    if not isinstance(X, np.ndarray):
        raise TypeError("Expected input of type np.ndarray, instead got {}".format(X.type))
    exps = np.exp(X - np.max(X))
    return (exps.T / np.expand_dims(np.sum(exps, axis=1), 1).T).T


def binxent(input:Tensor, labels:Tensor) -> Tensor:
    r''' Binary Cross Entopy, similar to nll but takes binary labels
                Args:
                    input (Tensor): a direct output of the network without softmax
                    target (Tensor: a tensor containing binary labels'''
    # input.shape == (batch_size, 1)
    # labels.shape == (batch_size, 1)
    raise NotImplementedError
    assert input.shape == labels.shape, 'binxent expects the same shape for labels and input'
    assert input.shape[1] == 1
    m = input.shape[0]
    # input probabilities
    p = stable_softmax(input.data)
    # binary labels (1 or 0)
    y = labels.data
    data = -(1/m) * np.sum(y*np.clip(np.log(p), -100, 100) + (1-y)*np.clip(np.log(1-p), -100, 100))
    requires_grad = input.requires_grad
    if requires_grad:
        def grad_fn(grad:np.ndarray) -> np.ndarray:
            assert np.min(p) > 0
            g = y/p - (1-y)/(1-p)
            return grad * g
        parent_nodes = [Node(input, grad_fn)]
    else:
        parent_nodes = []
    return Tensor(data, requires_grad, parent_nodes) 

def BCELoss(input:Tensor, labels:Tensor) -> Tensor:
    assert input.shape == labels.shape
    ones = Tensor(np.ones(input.shape))
    return (labels * clipped_log(input) + (ones - labels) * (ones - clipped_log(input))).sum() / Tensor(input.shape[0])
    
def log(t1:Tensor) -> Tensor:
    return _log(t1)

def clipped_log(t1:Tensor, clip=100) -> Tensor:
    data = np.clip(np.log(t1.data), -clip, clip)
    requires_grad = t1.requires_grad
    if requires_grad:
        parent_nodes = [Node(t1, lambda grad: np.clip(grad / t1.data, -100, 100))]
    else:
        parent_nodes = []

    return Tensor(data, requires_grad, parent_nodes)

def one_hot_encode(t1:Tensor, num_of_classes:int = None, 
                    dtype:Optional[Union[float, int]] = int, squeeze:bool = True) -> Tensor:
    r'''A utility function that takes a tensor and returns a one hot encoding
            (note: this function should be used on non gradient tracking tensors only
            
        Args: 
            t1 (Tensor): a n-tensor containing the data that should be encoded into a n+1-tensor
            num_of_classes (int): the number of classes to encode
            dtype (float, int): output data type
            squeeze (bool): Whether you want the output to be squeezed (will be depreciated)
            
        Returns:
            A tensor containing the encoded data (usually n+1-Tensor)'''

    # t1.shape == (num_of_batches, batch_size,) or t1.shape == (batch_size)
    # data.shape == (num_of_batches, batch_size, num_of_values)
    # @TODO clean up this code it is amazingly sloppy
    a = t1.data.astype(int)
    a = a.squeeze()
    if a.ndim > 1: raise ValueError("Input Tensor has more than one working dimensions, shape: {}".format(a.shape))
    if num_of_classes is None: num_of_classes = a.max() + 1
    
    assert a.ndim == 1 or a.ndim == 0, f'only accepts 1 or 0 tensors, got {a.ndim} dim tensor' + f' {a.shape}'
    data = np.zeros((a.size, num_of_classes))
    data[np.arange(a.size), a] = 1
    if squeeze: data = data.squeeze()
    return Tensor(data.astype(dtype), requires_grad=False)

def dropout(t1:Tensor, rate:float) -> Tensor:
    r''' Utility function that randomly sets values in a tensor to zero '''
    data = np.random.binomial()
    print(data)