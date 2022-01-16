import numpy as rnp
import autograd.numpy as np
### https://github.com/HIPS/autograd/blob/master/docs/tutorial.md
from autograd import grad, elementwise_grad
from nn.activations import *

# base implementation
class Layer:
    def __init__(self):
        self.input:rnp.ndarray = None
        self.output:rnp.ndarray = None

    def forward(self, x:np.ndarray) -> np.ndarray: raise NotImplementedError
    
    def backward(self, error, lr): raise NotImplementedError

# aka fully connected layer. It is only called linear because that is what pytorch calls it (pytorch ftw)
class Linear(Layer):
    def __init__(self, in_features:int, out_features:int, bias = True) -> None:
        self.in_features:int = in_features
        self.out_features:int = out_features
        try:
            assert isinstance(self.in_features, int) and isinstance(self.out_features, int)
        except AssertionError:
            raise NameError("feature count must be an integer, not a ", type(in_features), type(out_features))
        if bias:
            self.bias:rnp.ndarray = np.zeros(out_features)
        self.weights:rnp.ndarray = np.random.uniform(-1, 1, size=(in_features, out_features))

    def forward(self, x):
        # x.shape = (self.in_features,) = x_i
        # y_j = b_j + sum/i(x_i*w_ij)
        self.input = x
        if self.bias is not None:
            self.output = np.add(np.dot(self.weights.T, x), self.bias)
        else:
            self.output = np.dot(self.weights.T, x)
        return self.output

    # computes dE/dW, dE/dB for a given output_error=dE/dY. Returns input_error=dE/dX.
    def backward(self, error, lr):
        w_err = np.dot(self.input.T, error)

        # update weights and biases
        self.weights -= lr * w_err
        if self.bias is not None:
            self.bias -= lr * error

        return np.dot(error, self.weights.T)

class Sigmoid(Layer):
    def __init__(self):
        # cache maybe
        self.function = logsig
        # o' = o(x)(1-o(x))
        self.df_dx = elementwise_grad(logsig)
    
    def forward(self, x):
        self.input = x
        self.output = self.function(x)
        return self.output

    # Returns input_error=dE/dX for a given output_error=dE/dY.
    # learning_rate is not used because there is no "learnable" parameters.
    def backward(self, error, lr):
        # sigmoid_prime: o' = 
        return self.df_dx(self.input) * error

class HyperbolicTangent(Layer):
    def __init__(self):
        self.function = tansig
        self.df_dx = elementwise_grad(tansig)

    def forward(self, x):
        self.input = x
        self.output = self.function(x)
        return self.output

    # Returns input_error=dE/dX for a given output_error=dE/dY.
    # learning_rate is not used because there is no "learnable" parameters.
    def backward(self, error, lr): 
        return self.df_dx(self.input) * error

