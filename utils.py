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