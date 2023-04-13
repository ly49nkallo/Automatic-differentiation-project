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
def display(h1, h2):
    #record state
    fig, ax = plt.subplots(4, len(h1))

    '''Visualize state timeseries'''
    for i, a in enumerate(ax[0,:]):
        a.set_title(f'State {i}')
        a.axis('off')
        if i == len(h1) - 1:
            a.imshow(h1[i])
            break
        a.imshow(h1[i].reshape(int(np.sqrt(h1[i].size)),int(np.sqrt(h1[i].size))))
    ''''''
    '''Visualize pertubation timeseries'''
    for i, a in enumerate(ax[1,:]):
        a.axis('off')
        if i == len(h2) - 1:
            a.imshow(h1[i])
            break
        a.imshow(h2[i].reshape(int(np.sqrt(h2[i].size)),int(np.sqrt(h2[i].size))))
    '''Visualize differences'''
    diff = [h2[i] - h1[i] for i in range(len(h1))]
    for i, a in enumerate(ax[2,:]):
        a.axis('off')
        image = diff[i]
        if i == len(h2) - 1:
            a.imshow(image)
            break
        a.imshow(image.reshape(int(np.sqrt(image.size)),int(np.sqrt(image.size))))
    ''''''
    '''Significance threshold'''
    std = [np.std(diff[i]) for i in range(len(diff))]
    for i, a in enumerate(ax[3,:]):
        a.axis('off')
        a.set_title(f"std: {int(std[i] * 1000)/1000}")
        significance_factor = 1.5
        image = (diff[i] > std[i] * significance_factor).astype(np.byte)
        if i == len(h2) - 1:
            a.imshow(image)
            break
        a.imshow(image.reshape(int(np.sqrt(image.size)),int(np.sqrt(image.size))))
    ''''''
    plt.show()

def display2(input_states, simulate):
    h1 = simulate(np.zeros(shape=(1,28*28), dtype=np.float32))
    fig, ax = plt.subplots(len(input_states), len(h1), squeeze=True)
    if len(input_states) == 1:
        ax = ax.reshape(1, -1)
    for input_idx, input_state in enumerate(input_states):
        h2 = simulate(input_state)
        diff = [h2[i] - h1[i] for i in range(len(h1))]
        '''Significance threshold'''
        std = [np.std(diff[i]) for i in range(len(diff))]
        for i, a in enumerate(ax[input_idx,:]):
            a.axis('off')
            a.set_title(f"std: {int(std[i] * 1000)/1000}")
            significance_factor = 1.5
            image = (diff[i] > std[i] * significance_factor).astype(np.byte)
            if i == len(h2) - 1:
                a.imshow(image)
                break
            a.imshow(image.reshape(int(np.sqrt(image.size)),int(np.sqrt(image.size))))
        ''''''
    
    plt.show()

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
    
    init_states = []
    size_of_flash = 5
    for i in range(3):
        init_state = np.zeros(shape=(1,28,28), dtype=np.float32)
        x, y = np.random.randint(0,28-size_of_flash), np.random.randint(0,28-size_of_flash)
        init_state[0,x:x+size_of_flash,y:y+size_of_flash] += 1
        init_state = init_state.reshape(1,-1)
        init_states.append(init_state)

    display2(init_states, simulate)

def config():
    font = {'family' : 'DejaVu Sans',
            'weight' : 'bold',
            'size'   : 8}
    import matplotlib
    matplotlib.rc('font', **font)
    cmaps = ['binary', 'gist_yarg', 'gist_gray', 'gray', 'bone',
             'pink', 'spring', 'summer', 'autumn', 'winter', 'cool',
             'Wistia', 'hot', 'afmhot', 'gist_heat', 'copper']
    sequential = ['viridis', 'plasma', 'inferno', 'magma', 'cividis']
    matplotlib.rc('image', cmap="gray")
if __name__ == '__main__':
    config()
    main()