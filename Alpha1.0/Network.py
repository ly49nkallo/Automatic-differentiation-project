import autograd.numpy as np
from autograd import elementwise_grad, grad
import nn.error_funcs as ef

class Net:
    # fully connected multi-layer perceptron

    def __init__(self, layers=[], ):
        
        # list of layer classes
        self.layers = layers
        self.loss = ef.mse
        #self.loss_prime = elementwise_grad(self.loss)
        self.loss_prime = ef.mse_prime

    def add(self, layer):
        self.layers.append(layer)

    def predict(self, input):
        samples = len(input)
        result = []
        for i in range(samples):
            output = input[i]
            for layer in self.layers:
                output = layer.forward(output)
            result.append(output)
        
        return result

    def train(self, x_train, y_train, epochs, lr=0.001): 
        samples = len(x_train)

        for i in range(epochs):
            err = 0
            for j in range(samples):
                # forward prop
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward(output)
                
                # for visuals only
                err += self.loss(y_train[j], output)

                # back propigation
                error  = self.loss_prime(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward(error, lr)
            err /= samples
            print('epoch %d/%d   error=%f' % (i+1, epochs, err))
    