import matplotlib.pyplot as plt
from tqdm import tqdm
import autograd as a
import autograd.utils as u
import autograd.functional as f
import numpy as np
import pickle
import os
from pathlib import Path
from mnist_nn import MNIST_MLP, Custom_MNIST_MLP # get definition(s)
'''INVESTIGATE CRITICALITY IN MULTI-LAYER PERCEPTRONS'''
    
def main():
    # visualize each activation at each layer'
    name = 'Custom_MNIST_MLP'
    with open(Path(f"{os.getcwd()}/saved_models/{name}.pkl"), 'rb') as inp:
        model =  pickle.load(inp)
    params = list(model.parameters())
    params = [p.data for p in params] # get just the data
    def simulate(init_state):
        history = [init_state]
        state = init_state
        for p in params:
            if p.ndim == 1:
                state = state + p
                state = f.stable_softmax(state)
                history.append(state.copy())
            elif p.ndim == 2:
                state = state @ p
        return history
    h1 = simulate(np.zeros(shape=(1,28*28), dtype=np.float32))
    init_state_2 = np.zeros(shape=(1,28*28), dtype=np.float32)
    init_state_2[0,0] += 1
    h2 = simulate(init_state_2)
    assert len(h1) == len(h2), 'Histories not same length'
    #record state
    fig, ax = plt.subplots(2, len(h1))

    '''Visualize state timeseries'''
    for i, a in enumerate(ax.flatten()[0:len(h1)]):
        a.axis('off')
        if i == len(h1) - 1:
            a.imshow(h1[i])
            break
        a.imshow(h1[i].reshape(int(np.sqrt(h1[i].size)),int(np.sqrt(h1[i].size))))
    ''''''
    '''Visualize differences'''
    for i, a in enumerate(ax.flatten()[len(h1):]):
        a.axis('off')
        image = h2[i] - h1[i]
        if i == len(h2) - 1:
            a.imshow(image)
            break
        a.imshow(image.reshape(int(np.sqrt(image.size)),int(np.sqrt(image.size))))
    plt.show()
    ''''''


if __name__ == '__main__':
    main()