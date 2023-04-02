import matplotlib.pyplot as plt
from tqdm import tqdm
import autograd as a
import autograd.utils as u
import autograd.functional as f
import numpy as np
import pickle
import os
from pathlib import Path
from mnist_nn import MNIST_MLP # get definition
'''INVESTIGATE CRITICALITY IN MULTI-LAYER PERCEPTRONS'''

class XOR(a.Module): 
    def __init__(self):
        super().__init__()
        self.linear1 = a.Linear(2,2)
        self.linear2 = a.Linear(2,1)
        self.act1, self.act2 = a.Sigmoid(), a.Sigmoid()
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.act1(x)
        x = self.linear2(x)
        x = self.act2(x)
        return x
    
def main():
    # visualize each activation at each layer'
    name = 'MNIST_MLP'
    with open(Path(f"{os.getcwd()}/saved_models/{name}.pkl"), 'rb') as inp:
        model =  pickle.load(inp)
    params = list(model.parameters())
    params = [p.data for p in params] # get just the data
    init_state = np.zeros(shape=(1,28*28))
    history = [init_state.copy()]
    state = init_state
    for p in params:
        if p.ndim == 1:
            state = state + p
            state = f.stable_softmax(state)
            history.append(state.copy())
        elif p.ndim == 2:
            state = state @ p
        #record state
        
    fig, ax = plt.subplots(1, len(history), sharey=True)
    for i, a in enumerate(ax.flatten()):
        a.imshow(history[i])
    plt.show()
    print(state)


if __name__ == '__main__':
    main()