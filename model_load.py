import pickle
from tkinter import W

from network import fc_network

def model_loader(filename):

    with open(filename, 'rb') as f:
        layers = pickle.load(f)
        
    new_network = fc_network(layers=layers)
    
    return new_network