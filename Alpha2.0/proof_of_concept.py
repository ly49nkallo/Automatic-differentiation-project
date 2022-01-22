from doctest import OutputChecker
import autograd.numpy as np
from autograd import grad, jacobian, tensor_jacobian_product, hessian_tensor_product
import numpy as npr
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1. / (1 + np.exp(-x))

# shape of fake weight matrix (out_features, in_features)
IN_FEATURES = 5
OUT_FEATURES = 6
weight = np.zeros((OUT_FEATURES,IN_FEATURES))
weight2 = np.zeros((OUT_FEATURES,IN_FEATURES))
bias = npr.random.uniform(-1, 1, size=(OUT_FEATURES,))
bias2 = np.zeros((OUT_FEATURES,IN_FEATURES))
inputs = np.ones((IN_FEATURES))
outputs = sigmoid(np.add(np.dot(inputs, weight.T), bias))
target = np.array([1]*OUT_FEATURES).astype(float)
print('inputs', inputs, inputs.shape)
print('outputs', outputs, outputs.shape)
print('target', target, target.shape)
def mse(outputs,target):
    mse = 0
    for o, t in zip(outputs, target):
        mse += 1/6 * (t - o) ** 2
    return -mse
_grad = grad(mse)
print('mse before', mse(outputs, target))
gradient = _grad(outputs, target)
print('gradient', gradient)
# train
history = []
print(weight)
for i in range(500):
    weight = np.array([np.add(b, gradient) for b in weight.T])
    weight = weight.transpose()
    bias = np.add(bias, gradient)
    outputs = sigmoid(np.add(np.dot(inputs,weight.T), bias))
    _grad = grad(mse)
    print('mse after', mse(outputs,target))
    gradient = _grad(outputs,target) * 1e-1
    #print('gradient', gradient)
    history.append(-mse(outputs,target))

'''
for i in range(500):
    weight = np.array([np.add(b, gradient) for b in weight.T])
    weight = weight.transpose()
    bias = np.add(bias, gradient)
    outputs = sigmoid(np.add(np.dot(inputs,weight.T), bias))
    _grad = grad(mse)
    print('mse after', mse(outputs,target))
    gradient = _grad(outputs,target) * 1e-3
    #print('gradient', gradient)
    history.append(-mse(outputs,target))
'''

print(weight)
print(bias)

plt.plot(range(len(history)), history)
plt.show()