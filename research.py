import matplotlib.pyplot as plt
from tqdm import tqdm
import autograd as a
import autograd.utils as u
import autograd.functional as f
import numpy as np
import pickle
import os
from pathlib import Path
from mnist_nn import MNIST_MLP, Custom_MNIST_MLP, Custom_MNIST_MLP_2 # get definition(s)
import tqdm
from datetime import datetime
    
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

def display2(input_states, simulate, _max=5):
    h1 = simulate(np.zeros(shape=(1,28*28), dtype=np.float32))
    fig, ax = plt.subplots(min(len(input_states), _max), len(h1), squeeze=True)
    all_fires = []
    if len(input_states) == 1:
        ax = ax.reshape(1, -1)
    for input_idx, input_state in tqdm.tqdm(enumerate(input_states)):
        h2 = simulate(input_state)
        diff = [h2[i] - h1[i] for i in range(len(h1))]
        '''Significance threshold'''
        std = [np.std(diff[i]) for i in range(len(diff))]
        significance_factor = 1.
        fire = [(diff[i] > std[i] * significance_factor).astype(np.byte) for i in range(len(diff))]
        if input_idx < _max:
            for i, a in enumerate(ax[input_idx,:]):
                a.axis('off')
                a.set_title(f"std: {int(std[i] * 1000)/1000}")
                image = fire[i]
                if i == len(h2) - 1:
                    a.imshow(image)
                    break
                a.imshow(image.reshape(int(np.sqrt(image.size)),int(np.sqrt(image.size))))
            ''''''
        all_fires.append(fire.copy())
    #plt.show()
    return all_fires

def main():
    # visualize each activation at each layer'
    name = 'Custom_MNIST_MLP_2'
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
    size_of_flash = 7
    iter_ = 1000
    for i in tqdm.tqdm(range(iter_)):
        init_state = np.zeros(shape=(1,28,28), dtype=np.float32)
        x, y = np.random.randint(0,28-size_of_flash), np.random.randint(0,28-size_of_flash)
        init_state[0,x:x+size_of_flash,y:y+size_of_flash] += 1
        init_state = init_state.reshape(1,-1)
        init_states.append(init_state)
    #Execute
    fire = display2(init_states, simulate)
    # get counts
    counts = []
    for i in range(len(fire)):
        for j in range(1,len(fire[0]) - 1):
            #ignore final state and first state
            counts.append(fire[i][j].sum())
    print(datetime.now())
    print("###IMPORTANT STATISTICS###")
    print('iterations', iter_)
    mean = np.mean(counts)
    print('mean', mean)
    print('median', np.median(counts))
    print('std', np.std(counts))
    print('ratio of activation starting', (size_of_flash**2) / (28**2))
    print('ratio of activation average', 
          mean / np.average(
            [s.size for s in simulate(np.zeros((1,28**2)))][1:-1]
            )
          )
    plt.figure()
    plt.hist(counts)
    plt.figure()
    counts = np.array(counts).reshape(len(fire), len(fire[0]) - 2).astype(np.float32)
    counts[:,0] /= 100
    counts[:,1:] /= 64
    plt.plot(np.mean(counts, axis=0))
    plt.show()

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