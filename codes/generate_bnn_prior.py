import pickle
import argparse
import yaml

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn

from scipy.interpolate import CubicSpline


import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.optim import Adam
from tqdm.auto import trange

from models import deconvolution



class BNN(PyroModule):

    def __init__(self, n_in, n_out, layers):
        super().__init__()
        
        self.n_layers = len(layers)
        self.layers = PyroModule[torch.nn.ModuleList]([
                                PyroModule[nn.Linear](n_in, n_out)
        for j in range(self.n_layers)
        ])  

        self.activations = []
        for ii, layer in enumerate(layers):

            # Scaling the weights, for gaussian n^(-1/2) and for cauchy n^-1
            if ii == self.n_layers-1:
                if layers[layer]['type'] == 'gaussian':
                    weight = layers[layer]['weight']*float(1/np.sqrt(n_out))
                else:
                    weight = layers[layer]['weight']*float(1/n_out)
            else:
                weight = layers[layer]['weight']
            bias = layers[layer]['bias']

            if layers[layer]['type'] == 'cauchy':
                self.layers[ii].weight = PyroSample(dist.Cauchy(0.,
                                                torch.tensor(weight)).expand([n_out, n_out]).to_event(2))
                self.layers[ii].bias = PyroSample(dist.Cauchy(0.,
                                                torch.tensor(bias)).expand([n_out]).to_event(1))
            elif layers[layer]['type'] == 'gaussian':
                self.layers[ii].weight = PyroSample(dist.Normal(0.,
                                                torch.tensor(weight)).expand([n_out, n_out]).to_event(2))    
                self.layers[ii].bias = PyroSample(dist.Normal(0.,
                                                torch.tensor(bias)).expand([n_out]).to_event(1))
            else:
                print('Invalid layer!')
            
            self.activations.append(layers[layer]['activation'])

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, t, A, y=None):
        #t = t.reshape(-1, 1)
        
        
        if self.activations[0] == 'tanh':
            x = self.tanh(self.layers[0](t))
        elif self.activations[0] == 'relu':
            x = self.relu(self.layers[0](t))
        else:
            x = self.layers[0](t)
            

        for ii in range(1, self.n_layers-1):
            if self.activations[ii] == 'tanh':
                x = self.tanh(self.layers[ii](x))
            elif self.activations[ii] == 'relu':
                x = self.relu(self.layers[ii](x))
            else:
                x = self.layers[ii](x)

        if self.activations[-1] == 'tanh':
            x = self.tanh(self.layers[-1](x))
        elif self.activations[-1] == 'relu':
            x = self.relu(self.layers[-1](x))
        else:
            x = self.layers[-1](x)
    
        y_hat = torch.matmul(A, x)
        sigma = pyro.sample("sigma", dist.Uniform(0.,
                                                torch.tensor(0.01)))
        with pyro.plate("data", n_y):
            obs = pyro.sample("obs", dist.Normal(y_hat, sigma), obs=y)
        return x


def prior(t):
    t = t.detach().numpy()
    weights1 = np.random.normal(0., 0.05/100/np.sqrt(200), (100,200))
    bias1 = np.random.normal(10., 0.05, (200,))

    weights2 = np.random.normal(0., 0.01/np.sqrt(100), (200,100))
    bias2 = np.random.normal(0., 0.1, (100,))

    weights3 = np.random.normal(1., 0.5, (100,100))
    bias3 = np.random.normal(0., 0.01, (100,))

    x = np.tanh(weights1.T @ t + bias1)
    x =  np.tanh(weights2.T @ x + bias2)
    x =  weights3.T @ x + bias3

    return x



def generate_bnn_realization_plot(t):
    # Generate prior realizations, A not used
    realizations = np.empty((len(t), 2))
    for ii in range(0, 2):
        realizations[:,ii] = prior(t)
    
    plt.plot(t, realizations)
    plt.savefig('realization.png')




if __name__ == '__main__':
    
    t = torch.linspace(-1, 1, 100)

    generate_bnn_realization_plot(t)
    

    
        