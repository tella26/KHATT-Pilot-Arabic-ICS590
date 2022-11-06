import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from torch.nn import Parameter
from collections import defaultdict
import numpy as np

class HopfieldNet(nn.Module):

    def __init__(self, 
                 in_features,
                 bias=True,
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
        e = -0.5 * x @ self.weight @ x
        if self.bias is not None:
            e -= self.bias @ x
        return e
        
    def _run(self, x, eps=1e-6):
        """Synchronousl update

        Args:
            x (torch.Tensor): inputs
            eps (float): Defaults to 1e-6.

        """
        e = self._energy(x)

        for _ in range(self.max_iter):

            x = torch.sign(
                F.linear(x, self.weight, self.bias) 
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
        
        self.W = W 

