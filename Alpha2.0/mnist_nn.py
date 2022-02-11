import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset

from autograd.tensor import Tensor
from autograd.optim import SGD
from autograd.module import Module, Linear
from autograd.activation import Sigmoid, Tanh, Softmax
from autograd.functional import minxent, log

def moving_average(a, n=3) :
    if not isinstance(a, np.ndarray): a = np.array(a)
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def main():
    history = []
    loader = Dataloader('mnist', 32)
    model = Mlp(32*28*28, 10)
    optimizer = SGD(model.parameters(), lr = 0.001)
    for batch_idx, (data, target) in enumerate(loader):
        data = Tensor(np.expand_dims(data.flatten(), 0), requires_grad = True)
        target = Tensor(target)
        #print(data.shape, target.shape)
        #print(batch_idx, data.shape, target)
        optimizer.zero_grad()
        output = model(data)
        loss = minxent(output, target, is_one_hot=False)
        history.append(loss.data.copy())
        loss.backward()
        optimizer.step()
        print(batch_idx, 'loss', loss.data)
        if batch_idx > (10000/32): break
    plt.plot(history)
    plt.plot(moving_average(history, n=10))
    plt.show()
'''Author: Ty Brennan'''

class Dataloader:
    '''A class that allows us to get batches of data from huggingface datsets'''
    def __init__(self, dataset:str,batch_size, transforms:list=list(), train=True):
        self.dataset = dataset.upper()
        self.batch_size = batch_size
        self.train = train
        assert isinstance(self.dataset, str)
        if self.dataset == 'MNIST':
            if train:
                self.pairs = load_dataset('mnist', split='train')
            else:
                self.pairs = load_dataset('mnist', split='test')

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index < len(self.pairs):
            batch = self.pairs[self.index:self.index+self.batch_size]
            self.index += self.batch_size
            # we want to return a tuple of two npArrays of shape 32x28x28 and 32x1
            if self.dataset == 'MNIST':
                # getdata() has been depreciated
                return np.array([np.array(b) / 255 for b in batch['image']]).reshape(-1, 28, 28), np.array(batch['label'])
            else:
                raise NameError(f'There is no database called {self.dataset}')
        else:
            raise StopIteration

class Mlp(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = Linear(in_features, 128)
        self.linear2 = Linear(128, 32)
        self.linear3 = Linear(32, out_features)
        self.act = Sigmoid()
        self.act2 = Sigmoid()
        self.softmax = Softmax()

    def forward(self, x):
        x = self.linear(x)
        x = self.act(x)
        x = self.linear2(x)
        x = self.act2(x)
        x = self.linear3(x)
        #x = self.act2(x)
        x = self.softmax(x)
        return x


if __name__ == '__main__':
    main()