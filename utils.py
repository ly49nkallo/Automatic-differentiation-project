from datasets import load_dataset
import numpy as np
import PIL

class Dataloader:
    def __init__(self, dataset:str,batch_size, transforms:list=list(), train=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.train = train
        assert isinstance(self.dataset, str)
        if dataset == 'MNIST':
            if train:
                self.pairs = load_dataset('mnist', split='train')
            else:
                self.pairs = load_dataset('mnist', split='test')

    def __iter__(self):
        self.index = - self.batch_size
        return self

    def __next__(self):
        if self.index < len(self.pairs):
            self.index += self.batch_size
            batch = self.pairs[self.index:self.index+self.batch_size]
            batch['image'] = np.array(list(map(lambda x: np.array(x.getdata()).reshape(28,28),batch['image'])))
            batch = np.array(list(zip(batch["image"], batch["label"])))
            print(batch[1,0])
            return batch
        else:
            raise StopIteration