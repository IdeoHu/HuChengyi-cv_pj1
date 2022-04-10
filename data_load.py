import struct
import numpy as np
import matplotlib.pyplot as plt

class MNIST():
    
    def __init__(self):
        
        np.random.seed(21)
        
        with open('MNIST//train-images.idx3-ubyte', 'rb') as figpath:
            magic, num, rows, cols = struct.unpack('>IIII', figpath.read(16))
            train_images = np.frombuffer(figpath.read(), dtype=np.uint8).reshape(-1, 28*28)
            
        with open('MNIST//train-labels.idx1-ubyte', 'rb') as labelpath:
            magic, n = struct.unpack('>II', labelpath.read(8))
            train_labels = np.frombuffer(labelpath.read(), dtype=np.uint8).reshape(-1, 1) + 1
            
        index = np.arange(num)
        np.random.shuffle(index)
        new_train_images, new_train_labels = train_images[index, :][:-10000, :], train_labels[index][:-10000]
        valid_images, valid_labels = train_images[index, :][-10000:, :], train_labels[index][-10000:]
        
        self.train_set = {'fig': new_train_images, 'label': new_train_labels, 'num': num-10000}
        self.valid_set = {'fig': valid_images, 'label': valid_labels, 'num': 10000}
        
        with open('MNIST//t10k-images.idx3-ubyte', 'rb') as figpath:
            magic, num, rows, cols = struct.unpack('>IIII', figpath.read(16))
            test_images = np.frombuffer(figpath.read(), dtype=np.uint8).reshape(-1, 28*28)
            
        with open('MNIST//t10k-labels.idx1-ubyte', 'rb') as labelpath:
            magic, n = struct.unpack('>II', labelpath.read(8))
            test_labels = np.frombuffer(labelpath.read(), dtype=np.uint8).reshape(-1, 1) + 1
            
        self.test_set = {'fig': test_images, 'label': test_labels, 'num': n}
        
        mean = np.mean(self.train_set['fig'])
        var = np.var(self.train_set['fig'])
        self.mean = mean
        self.var = var
        d = np.sqrt(var)
        self.train_set['fig'] = (self.train_set['fig'] - mean) / d
        self.valid_set['fig'] = (self.valid_set['fig'] - mean) / d
        self.test_set['fig'] = (self.test_set['fig'] - mean) / d
        
    def get_one_data(self):
        
        num = self.train_set['num']
        index = np.random.randint(0, num)
        
        sample = self.train_set['fig'][index:index + 1, :].copy().T
        label = self.train_set['label'][index, 0]
        
        return sample, label
    
    def visualize(self, one_sample):
        one_sample = (one_sample.T * np.sqrt(self.var) + self.mean).T.astype(np.uint8)
        fig = one_sample.reshape(28, 28)
        plt.imshow(fig)
        plt.show()


if __name__ == '__main__':
    dataset = MNIST()
    sample, label = dataset.get_one_data()
    dataset.visualize(sample)
    print(label)