import pickle
from copy import deepcopy
from math import pi

import numpy as np
from matplotlib import pyplot as plt

def log_loss(pred, label):
    temp = 1 - pred
    temp[label-1, 0] = 1 - temp[label-1, 0]  
    val = -np.sum(np.log(temp))
    temp[label-1, 0] = -temp[label-1, 0] 
    grad = 1 / temp
    return val, grad

class fc_network():
    def __init__(self, layers):
        self.layers = layers
        self.error = []
    
    def train(self, max_iter, lr, dataset, loss='log_loss', lamb=0):

        x = []
        t_loss = []
        accuracy = []

        for iter in range(max_iter): 

            x_in, x_label = dataset.get_one_data() 

            for layer in self.layers.values():
                x_in = layer.forward(x_in)
            yhat = x_in

            f, g_in = log_loss(yhat, x_label)
            for layer in list(self.layers.values())[::-1]:
                g_in = layer.backward(g_in)

            lr_schedule = lr * (1 + np.cos(pi * iter/ max_iter)) / 2
            for layer in self.layers.values():
                layer.step(lr_schedule, lamb)

            if iter % 10000 == 0 and iter != 0:
                error, loss = self.predict(dataset, 'valid')
                self.error.append(error)
                print('\niters: %5d' % iter, sep=', ')
                print('error rate: %.5f' % error)
                print('average loss: %.5f' % loss)
                x.append(iter)
                t_loss.append(loss)
                accuracy.append(1-error)
                
        try:
            train_loss_lines.remove(train_loss_lines[0])
            train_accuracy_lines.remove(train_loss_lines[0])
        except Exception:
            pass
        train_loss_lines = plt.plot(x, t_loss, label = 'loss', color = 'r')
        plt.savefig('loss.jpg')
        plt.show()
        train_accuracy_lines = plt.plot(x, accuracy, label = 'accuracy', color = 'b')
        plt.savefig('accuracy.jpg')
        plt.show()

        error, _ = self.predict(dataset, 'valid')
        print('\nerror rate: %.5f' % error)
        
        return error
    
    def predict(self, dataset, type):
        if type == 'valid':
            data = dataset.valid_set['fig']
            label = dataset.valid_set['label']
            num = dataset.valid_set['num']
        elif type == 'test':
            data = dataset.test_set['fig']
            label = dataset.test_set['label']
            num = dataset.test_set['num']
        
        x_in = data.T

        for layer in self.layers.values():
            x_in = layer.forward(x_in, False)  # train=False
        yhat = x_in
        
        pred = (np.argmax(yhat, axis=0).T + 1).reshape((-1, 1))
        error = np.sum((pred - label) != 0) / label.shape[0]
        
        loss = 0
        for i in range(num):
            loss += log_loss(yhat[:, i:i+1], label[i, 0])[0]
        loss = loss / num

        return error, loss

    def model_save(self, filename):
        
        print('\nModel saving.')
        layers = deepcopy(self.layers) 
        with open(filename, 'wb') as f:
            pickle.dump(layers, f)
        print('\nComplete.')
