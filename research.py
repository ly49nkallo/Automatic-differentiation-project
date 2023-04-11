import matplotlib.pyplot as plt
from tqdm import tqdm
import autograd as a
import autograd.utils as u
import autograd.functional as f
import numpy as np
import matplotlib.pyplot as plt
'''INVESTIGATE CRITICALITY IN MULTI-LAYER PERCEPTRONS'''
    
def main():
    model = u.load_model(name="MNIST_MLP")
    parameters = [p.data for p in model.parameters()]
    input_state = np.zeros(28**2)
    input_state[28**2//2:] = 1
    input_state = a.Tensor(input_state.reshape((-1,28**2)))
    # plt.imshow(input_state.data.reshape((28,28)))
    print(model(input_state).data)

    '''Test many different pertubations in the network'''
    state = input_state.data
    weights = []
    for p in model.parameters():
        if p.ndim == 2: weights.append(p.data)
    fig, axs = plt.subplots(1,len(weights), squeeze=True)
    for i, ax in enumerate(axs):
        ax.imshow(weights[i].squeeze())
    # print(state)
    plt.show()

    


if __name__ == '__main__':
    main()