import numpy as np
import matplotlib.pyplot as plt

from autograd.tensor import Tensor
from autograd.functional import minxent, binxent, mse

def test_simple_minxent():

    history = []
    t1 = Tensor([[1,1,1] for _ in range(3)], requires_grad=True)
    labels = Tensor([[1, 0, 0] for _ in range(3)])

    #assert np.abs(np.sum(t2.data) - 1.) < 1e-3 # there is still some numerical instability
    
    for _ in range(50):
        t1 = Tensor(t1.data - (t1.grad.data * 0.1), requires_grad=True)
        print(t1.data)
        t2 = t1.softmax()
        print(t2.data)
        t3 = minxent(t2, labels)
        history.append((0, t3.data.copy()))
        t3.backward()

        print(t3.data)
        
        #print(t1.grad.data)
        #print(t2.grad.data)

    for _ in range(50):
        t1 = Tensor(t1.data - (t1.grad.data * 0.01), requires_grad=True)
        print('t1',t1.data)
        t2 = t1.softmax()
        print('t2', t2.data)
        t3 = minxent(t2, labels)
        history.append((0, t3.data.copy()))
        t3.backward()

        print('t3',t3.data)
        
        print('t1 grad', t1.grad.data)
        print('t2 grad', t2.grad.data)
    
    plt.plot(history)
    plt.show()

test_simple_minxent()