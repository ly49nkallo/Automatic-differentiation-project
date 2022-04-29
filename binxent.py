import numpy as np
import matplotlib.pyplot as plt

from autograd.tensor import Tensor
from autograd.functional import minxent, binxent, mse, softmax, nll, BCELoss

def test_simple_minxent():

    history = []
    t1 = Tensor(np.ones((16,1)), requires_grad=True)
    labels = Tensor(np.zeros((16,1)))

    #assert np.abs(np.sum(t2.data) - 1.) < 1e-3 # there is still some numerical instability
    
    for _ in range(100):
        t1 = Tensor(t1.data - (t1.grad.data * 0.3), requires_grad=True)
        t3 = BCELoss(t1, labels)
        print(t3.data)
        history.append((0, t3.data.copy()))
        t3.backward()

        
        
        #print(t1.grad.data)
        #print(t2.grad.data)
    
    plt.plot(history)
    plt.show()

test_simple_minxent()