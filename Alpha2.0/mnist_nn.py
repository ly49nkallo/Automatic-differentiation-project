import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset

from autograd.tensor import Tensor
from autograd.optim import SGD
from autograd.module import Module, Linear
from autograd.activation import Sigmoid, Tanh, Softmax

def main():
    loader = Dataloader('mnist', 4)
    for idx, (data, labels) in enumerate(loader):
        print(idx, data, labels)
        if idx > 10: break
'''Author: Ty Brennan'''

from datasets import load_dataset
import numpy as np

class Dataloader:
    '''A class that allows us to get batches of data from huggingface datsets'''
    def __init__(self, dataset:str,batch_size, transforms:list=list(), train=True):
        self.dataset = dataset.upper()
        self.batch_size = batch_size
        self.train = train
        assert isinstance(self.dataset, str)
        if dataset == 'MNIST':
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
                return np.array([b.getdata() / 255 for b in batch['image']]).reshape(-1, 28, 28), np.array(batch['label'])
            else:
                raise NameError(f'There is no database called {self.dataset}')
        else:
            raise StopIteration

class model(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = Linear(in_features, 5)
        self.linear2 = Linear(5, out_features)
        self.act = Sigmoid()
        self.act2 = Sigmoid()
        self.softmax = Softmax()

    def forward(self, x):
        x = self.linear(x)
        x = self.act(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x


if __name__ == '__main__':
    main()