from autograd.tensor import Tensor, Dependency
#import autograd
#Tensor = autograd.tensor.Tensor
import numpy as np

def tanh(t:Tensor) -> Tensor:
    data = (np.exp(t.data) - np.exp(-t.data) / np.exp(t.data) + np.exp(-t.data))
    requires_grad = t.requires_grad
    if requires_grad:
        def grad_fn(grad:np.ndarray) -> np.ndarray:
            return grad * (1- (data * data))
        depends_on = [Dependency(t, grad_fn)]
    else:
        depends_on = []
    
    return Tensor(data, requires_grad, depends_on)

def relu(t:Tensor) -> Tensor:
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

def identity(t:Tensor) -> Tensor:
    return Tensor(t.data, t.requires_grad, [Dependency(t, lambda x: x)])

def softmax(t:Tensor) -> Tensor:
    data = np.exp(t.data) / (np.sum(np.exp(t.data)))

def mse(output:Tensor, labels:Tensor) -> Tensor: 
    return ((labels - output) ** 2).sum() / Tensor(labels.size())

# also called cross entropy loss due to it's usage by statistical analysis
# https://gombru.github.io/assets/cross_entropy_loss/intro.png
def negative_log_likelihood(output:Tensor, labels:Tensor) -> Tensor:
    r'''AKA Cross entropy loss by statastitians
        Args:
            output (Tensor): the input tensor (preferably softmaxed)
            labels (Tensor): a tensor containing the ground truth (preferably one-hot vector)'''
    raise NotImplementedError('Still need to implement tensor_log')
    