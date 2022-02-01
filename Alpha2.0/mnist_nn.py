import numpy as np
import matplotlib.pyplot as plt

from autograd.tensor import Tensor
from autograd.optim import SGD

def main():
    a = Tensor(np.ones(5), requires_grad=True)
    b = a.softmax()
    #print(b)
    c = b.sum()
    c.backward()
    print('', a, '\n', b, '\n', c)
    print()
    print('', a.grad, '\n', b.grad, '\n', c.grad)

if __name__ == '__main__':
    main()
