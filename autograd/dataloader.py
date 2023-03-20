from datasets import load_dataset
import numpy as np

def dummy_msg(dset:str): return f"Using dummy dataset for {dset}"
class Dataloader:
    '''A class that allows us to get batches of data from huggingface datsets'''
    def __init__(self, dataset:str,batch_size:int, 
                transforms:list=list(), 
                train:bool=True,
                shuffle:bool=False,
                dummy:bool=False):
        self.dataset = dataset.upper()
        self.batch_size = batch_size
        self.train = train
        assert isinstance(self.dataset, str) and isinstance(self.batch_size, int)
        if self.dataset == 'MNIST':
            # Data source
            # https://huggingface.co/datasets/mnist
            if train:
                #shape is image:(60000,28,28) label: (60000)
                if not dummy:
                    self.dset = load_dataset('mnist', split='train')
                else:
                    print(dummy_msg(self.dataset.lower().strip()))
                    image = np.clip(np.random.randn(60000,28,28) * 255, 0, 255)
                    label = np.random.randint(0,10,60000)
                    self.dset = {'image':image, 'label':label}
            else:
                if not dummy:
                    self.dset = load_dataset('mnist', split='test')
                else:
                    print(dummy_msg(self.dataset.lower().strip()))
                    image = np.clip(np.random.randn(10000,28,28) * 255, 0, 255)
                    label = np.random.randint(0,10,10000)
                    self.dset = {'image':image, 'label':label}
            self.data = self.dset['image']
            self.label = self.dset['label']
            del self.dset
            self.pairs = list(zip(self.data, self.label))
            if shuffle:
                np.random.seed(123)
                np.random.shuffle(self.pairs)
            self.data, self.label = zip(*self.pairs)
            self.data = np.array(list(map(lambda img: np.array(img) / 255, self.data))).reshape(-1, 28, 28)
            self.label = np.array(self.label).reshape(-1)
        
        elif self.dataset == 'NOT':
            if train:
                self.data = (np.random.rand(1000, 100) > 0.5).astype(np.int32)
                self.label = -self.data.copy() + 1
            else:
                self.data = (np.random.rand(100, 100) > 0.5).astype(np.int32)
                self.label = -self.data.copy() + 1
        
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
