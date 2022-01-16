import autograd.numpy as np

# mse(y,z) = mean((z-y)**2)
def mse(y_true, y_pred):
    return np.mean(np.power(y_true-y_pred, 2))
