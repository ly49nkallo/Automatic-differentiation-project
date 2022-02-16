from datasets import load_dataset
import numpy as np

class Dataloader:
    '''A class that allows us to get batches of data from huggingface datsets'''
    def __init__(self, dataset:str,batch_size, 
                transforms:list=list(), 
                train=True,
                shuffle=False):
        self.dataset = dataset.upper()
        self.batch_size = batch_size
        self.train = train
        assert isinstance(self.dataset, str)
        if self.dataset == 'MNIST':
            if train:
                self.dset = load_dataset('mnist', split='train')
            else:
                self.dset = load_dataset('mnist', split='test')
            self.data = self.dset['image']
            self.label = self.dset['label']
            del self.dset
            self.pairs = list(zip(self.data, self.label))
            if shuffle:
                np.random.seed(69420)
                np.random.shuffle(self.pairs)
            self.data, self.label = zip(*self.pairs)
            self.data = np.array(list(map(lambda img: np.array(img) / 255, self.data))).reshape(-1, 28, 28)
            self.label = np.array(self.label).reshape(-1)
        
        
    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index < len(self.pairs) - 1:
            batch_data = self.data[self.index:self.index+self.batch_size]
            batch_label = self.label[self.index:self.index+self.batch_size]
            self.index += self.batch_size
            # we want to return a tuple of two npArrays of shape 32x28x28 and 32x1
            # getdata() has been depreciated
            return batch_data, batch_label
        else:
            raise StopIteration

    def __len__(self):
        return len(self.pairs) // self.batch_size