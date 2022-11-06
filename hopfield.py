import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from torch.nn import Parameter
from collections import defaultdict
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm

class HopfieldNet(nn.Module):

    def __init__(self, 
                 in_features,
                 bias=False,
                 threshold=0.0, 
                 max_iter=128):
        super(HopfieldNet, self).__init__()

        self.in_features = in_features
        self.threshold = threshold
        self.max_iter = max_iter

        self.weight = Parameter(torch.zeros(in_features, in_features), requires_grad=False)
        if bias:
            self.bias = Parameter(torch.Tensor(in_features), requires_grad=False)
        else:
            self.register_parameter('bias', None)

    def _energy(self, x):
        bias = self.bias 
        e = -0.5 * x @ self.W @ x
        #bias = bias.to(dtype=torch.float64)
        if bias is not None:
            e -= bias @ x
        return e
        
    def _run(self, x, eps=1e-6):
        """Synchronous update

        Args:
            x (torch.Tensor): inputs
            eps (float): Defaults to 1e-6.

        """
        x = torch.tensor(x,dtype=float)
        e = self._energy(x)

        for _ in range(self.max_iter):

            x = torch.sign(
                F.linear(x, self.W, self.bias) 
                    - self.threshold)

            new_e = self._energy(x)
            if abs(new_e - e) < eps:
                return x

            e = new_e
        return x

    def forward(self, x):
        assert x.ndim == 1
        
        return self._run(x)
    
    def extra_repr(self):
        return 'in_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
    def train_weights(self, train_data):
        print("Start to train weights...")
        num_data =  len(train_data)
        # data_train = np.array(train_data)
        self.num_neuron = train_data[0].shape[0]
        
        # initialize weights
        W = np.zeros((self.num_neuron, self.num_neuron))
        rho = np.sum([np.sum(t) for t in train_data]) / (num_data*self.num_neuron)
        
        # Hebb rule
        for i in tqdm(range(num_data)):
            t = train_data[i] - rho
            W += np.outer(t, t)
        
        # Make diagonal element of W into 0
        diagW = np.diag(np.diag(W))
        W = W - diagW
        W /= num_data
        W = torch.tensor(W,dtype=float)
        self.W = W 
        
    def predict(self, data, num_iter=20, threshold=0, asyn=False):
        print("Start to predict...")
        self.num_iter = num_iter
        self.threshold = threshold
        self.asyn = asyn
        
        # Copy to avoid call by reference 
        copied_data = np.copy(data)
        
        # Define predict list
        predicted = []
        for i in tqdm(range(len(data))):
            predicted.append(self._run(copied_data[i]))
        return predicted
    
    def plot_weights(self):
        plt.figure(figsize=(6, 5))
        w_mat = plt.imshow(self.W, cmap=cm.coolwarm)
        plt.colorbar(w_mat)
        plt.title("Network Weights")
        plt.tight_layout()
        plt.savefig("weights.png")
        plt.show()

