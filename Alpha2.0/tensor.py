import numpy as np

class Tensor(np.ndarray):
    def __init__(self):
        super(Tensor, self).__init__()
        self.grad = None
        self.requires_grad = True
