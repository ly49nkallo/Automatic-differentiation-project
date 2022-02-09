from autograd.tensor import Tensor, Dependency, _log
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
    raise NotImplementedError()

def mse(output:Tensor, labels:Tensor) -> Tensor: 
    return ((labels - output) ** 2).sum() / Tensor(labels.size())

# also called cross entropy loss due to it's usage by statistical analysis (minxent)
# https://gombru.github.io/assets/cross_entropy_loss/intro.png
def minxent(X:Tensor, y:Tensor, is_one_hot = True) -> Tensor:
    r'''AKA Categorical Cross entropy loss by statastitians or negative log likelihood (NLL)
        Args:
            output (Tensor): the input tensor (preferably softmaxed)
            label (Tensor): a tensor containing the ground truth (preferably one-hot vector)'''
    
    # output.shape (batch_size, num_classes)
    # labels.shape (batch_size, )
    #m = y.shape[0]
    m = y.shape[0]
    if not is_one_hot:
        y = one_hot_encode(y)
    return -((log(X) * y).sum()) / Tensor(m)


def binxent(output:Tensor, labels:Tensor) -> Tensor:
    r''' Binary Cross Entopy, takes binary labels'''
    raise NotImplementedError()
    
def log(t1:Tensor) -> Tensor:
    return _log(t1)

def one_hot_encode(t1:Tensor) -> Tensor:
    r''' A utility function that takes a tensor and returns a one hot encoding
            (note: this function should be used on non gradient tracking tensors only'''

    # t1.shape == (num_of_batches, batch_size,)
    # data.shape == (num_of_batches, batch_size, num_of_values)
    # @TODO clean up this code it is amazingly sloppy
    data = t1.data
    if data.ndim == 1:
        data = np.expand_dims(data, 0)
    assert data.ndim == 2
    data = np.zeros(list(data.shape) + [t1.data.max() + 1])
    assert data.ndim == 3, data.shape
    data[np.arange(data.shape[0]), np.arange(data.shape[1]), t1.data] = 1
    print(data.shape)
    if data.shape[0] == 1:
        data = data.squeeze(axis = 0)
    
    return Tensor(data)