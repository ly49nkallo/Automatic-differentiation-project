from autograd.tensor import Tensor, Node, _log
#import autograd
#Tensor = autograd.tensor.Tensor
import numpy as np
from typing import Optional, Union

def tanh(t:Tensor) -> Tensor:
    data = (np.exp(t.data) - np.exp(-t.data) / np.exp(t.data) + np.exp(-t.data))
    requires_grad = t.requires_grad
    if requires_grad:
        def grad_fn(grad:np.ndarray) -> np.ndarray:
            return grad * (1- (data * data))
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
            input (Tensor): the input tensor (preferably softmaxed)
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
                target (Tensor: a tensor containing labels
                
            Note: target shape's first dimention must be dimension num_classes'''
    # https://deepnotes.io/softmax-crossentropy
    # input.shape == (batch_size, num_classes)
    # target.shape == (num_classes, 1)
    m = target.shape[0]
    p = stable_softmax(input.data)
    #assert target.data.ndim == 2
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
    exps = np.exp(X - np.max(X))
    return (exps.T / np.expand_dims(np.sum(exps, axis=1), 1).T).T


def binxent(output:Tensor, labels:Tensor) -> Tensor:
    r''' Binary Cross Entopy, takes binary labels'''
    raise NotImplementedError()
    
def log(t1:Tensor) -> Tensor:
    return _log(t1)

def one_hot_encode(t1:Tensor, num_of_classes:int = None, dtype:Optional[Union[float, int]] = int, squeeze:bool = True) -> Tensor:
    r''' A utility function that takes a tensor and returns a one hot encoding
            (note: this function should be used on non gradient tracking tensors only'''

    # t1.shape == (num_of_batches, batch_size,)
    # data.shape == (num_of_batches, batch_size, num_of_values)
    # @TODO clean up this code it is amazingly sloppy
    
    a = t1.data.astype(int)
    if num_of_classes is None: num_of_classes = a.max() + 1
    #assert num_of_classes == 10
    a = a.squeeze()
    assert a.ndim == 1 or a.ndim == 0, f'only accepts 1 or 0 tensors, got {a.ndim} dim tensor' + f' {a.shape}'
    data = np.zeros((a.size, num_of_classes))
    data[np.arange(a.size), a] = 1
    if squeeze: data = data.squeeze()
    return Tensor(data.astype(dtype))