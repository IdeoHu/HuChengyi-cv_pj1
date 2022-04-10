import numpy as np
from scipy.signal import convolve2d


class fc_layer():
    
    def __init__(self, in_dim, out_dim, sigma, bias, drop):
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        np.random.seed(21)        
        self.init_weights(sigma)
        
        self.bias = bias
        self.drop = drop
        self.momentum_W = np.zeros((out_dim, in_dim))
        self.momentum_b = np.zeros((out_dim, 1))
    
    def init_weights(self, sigma):
        # Weights.
        self.W = np.random.normal(0, sigma, size=(self.out_dim, self.in_dim))
        # Bias.
        self.b = np.zeros((self.out_dim, 1))
        
    def forward(self, x, train=True):
        if train:
            rnd = np.random.random(x.shape)    
            mask = rnd <= self.drop
            x = x * mask
        else:
            x = x * self.drop
        self.x = x
        if self.bias:
            return np.dot(self.W, x) + np.tile(self.b, x.shape[1])
        else:  
            return np.dot(self.W, x)
    
    def backward(self, grad):
        x = self.x
        self.grad_W = np.dot(grad, x.T)
        if self.bias:
            self.grad_b = np.sum(grad, axis=1).reshape(-1, 1)
        return np.dot(self.W.T, grad)
    
    def step(self, lr, lamb=0):
        
        self.W = self.W - lr * (self.grad_W + lamb * self.W)
        if self.bias:
            self.b = self.b - lr * self.grad_b

class activate_layer():
    
    def __init__(self, type):
        self.type = type
        
    def forward(self, x, *_):
        self.x = x
        if self.type == 'sigmoid':
            self.output = 1 / (np.exp(-x) + 1)
        elif self.type == 'tanh':
            self.output = (1 - np.exp(-2*x)) / (1 + np.exp(-2*x))
        elif self.type == 'relu':
            self.output = x * (x > 0)
        return self.output
    
    def backward(self, grad):
        a = self.output
        if self.type == 'sigmoid':
            return a * (1-a) * grad
        elif self.type == 'tanh':
            return (1 - a**2) * grad
        elif self.type == 'relu':
            return (a > 0) * grad
        
    def step(self, *_):
        pass
        
    
class softmax_layer():
    
    def __init__(self):
        pass
    
    def forward(self, x, *_):
        self.x = x
        exp_x = np.exp(x)
        self.output = exp_x / np.sum(exp_x, axis = 0)
        return self.output
    
    def backward(self, grad):
        a = self.output
        A = -np.dot(a, a.T)
        A = A + np.diag(a.reshape((-1,)))
        return np.dot(A.T, grad)
    
    def step(self, *_):
        pass

class pooling_layer():
    
    def __init__(self, stride):
        self.stride = stride
        
    def forward(self, X):
        pass