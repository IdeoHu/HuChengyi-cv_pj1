import pickle

import numpy as np

from data_load import MNIST
from layers import activate_layer, fc_layer, softmax_layer
from model_load import model_loader
from network import fc_network

if __name__ == '__main__':
    
    dataset = MNIST()
    
    lr_set = [1e-3, 5e-4, 1e-4]
    lamb_set = [0.1, 0.01, 0.001]
    neural_set = [100, 300, 500]
    
   # network = fc_network(
   #     layers = {
   #         'fc1': fc_layer(in_dim=784, out_dim=neural[i], sigma=np.sqrt(2/(784+neural[i])), bias=True, drop=1),
   #         'activ1': activate_layer(type='relu'),
   #         'fc2': fc_layer(in_dim=neural[i], out_dim=10, sigma=np.sqrt(2/(10+neural[i])), bias=True, drop=1),
   #         'softmax': softmax_layer()
   #         }
   # )    

    network = fc_network(
        layers = {
            'fc1': fc_layer(in_dim=784, out_dim=300, sigma=np.sqrt(2/(784+300)), bias=True, drop=1),
            'activ1': activate_layer(type='relu'),
            'fc2': fc_layer(in_dim=300, out_dim=10, sigma=np.sqrt(2/(10+300)), bias=True, drop=1),
            'softmax': softmax_layer()
            }
    )    
    
    error = network.train(max_iter=300000, lr=1e-3, dataset=dataset, loss='log_loss', lamb=0.0001) 
  # error = network.train(max_iter=300000, lr=lr_set[i], dataset=dataset, loss='log_loss', lamb=lamb_set[i])

    network.model_save('net_model.pkl')   
    new_net_work = model_loader('net_model.pkl')
    print(new_net_work.predict(dataset, 'test'))
