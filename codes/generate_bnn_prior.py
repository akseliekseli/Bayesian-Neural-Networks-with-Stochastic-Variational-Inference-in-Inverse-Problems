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

        self.layers = PyroModule[torch.nn.ModuleList]([])
        self.activations = []
        for ii, layer in enumerate(layers):
            # Creating and appending torch layer
            if ii == 0:
                self.layers.append(PyroModule[nn.Linear](n_in, layers[layer]['layer_size']))
            elif ii == self.n_layers-1:
                input = self.layers[-1].out_features
                self.layers.append(PyroModule[nn.Linear](input, n_out))
            else:
                input = self.layers[-1].out_features
                self.layers.append(PyroModule[nn.Linear](input, layers[layer]['layer_size']))


            # Scaling the weights, for Gaussian n^(-1/2) and for Cauchy n^-1
            if ii != 0:
                if layers[layer]['type'] == 'gaussian':
                    weight_scale = float(1 / np.sqrt(self.layers[ii].in_features))
                elif layers[layer]['type'] == 'cauchy':
                    weight_scale = float(1 / self.layers[ii].in_features)
            else:
                weight_scale = 1.0

            weight = float(layers[layer]['weight'] * weight_scale)
            bias = float(layers[layer]['bias'])

            if layers[layer]['type'] == 'cauchy':
                if ii == 0:  # First layer
                    self.layers[ii].weight = PyroSample(dist.Cauchy(0., torch.tensor(weight)).expand([self.layers[ii].out_features, 1]).to_event(2))
                elif ii == self.n_layers-1:
                    self.layers[ii].weight = PyroSample(dist.Cauchy(0., torch.tensor(weight)).expand([1, self.layers[ii].in_features]).to_event(2))
                else:
                    self.layers[ii].weight = PyroSample(dist.Cauchy(0., torch.tensor(weight)).expand([self.layers[ii].out_features, self.layers[ii].in_features]).to_event(2))

                if ii == self.n_layers - 1:  # Special case for the last layer
                    self.layers[ii].bias = PyroSample(dist.Cauchy(0., torch.tensor(bias)).expand([1, 1]).to_event(2))
                    #pass
                else:
                    self.layers[ii].bias = PyroSample(dist.Cauchy(0., torch.tensor(bias)).expand([1, self.layers[ii].out_features]).to_event(2))
                
            elif layers[layer]['type'] == 'gaussian':
                if ii == 0:  # First layer
                    self.layers[ii].weight = PyroSample(dist.Normal(0., torch.tensor(weight)).expand([self.layers[ii].out_features, 1]).to_event(2))
                elif ii == self.n_layers-1:
                    self.layers[ii].weight = PyroSample(dist.Normal(0., torch.tensor(weight)).expand([1, self.layers[ii].in_features]).to_event(2))
                else:
                    self.layers[ii].weight = PyroSample(dist.Normal(0., torch.tensor(weight)).expand([self.layers[ii].out_features, self.layers[ii].in_features]).to_event(2))

                if ii == self.n_layers - 1:  # Special case for the last layer
                    self.layers[ii].bias = PyroSample(dist.Normal(0., torch.tensor(bias)).expand([1, 1]).to_event(2))
                    #pass
                else:
                    self.layers[ii].bias = PyroSample(dist.Normal(0., torch.tensor(bias)).expand([1, self.layers[ii].out_features]).to_event(2))
            else:
                print('Invalid layer!')

            self.activations.append(layers[layer]['activation'])

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, t, A, y=None):
        t = t.reshape(-1, 1)

        for ii in range(self.n_layers):
            if self.activations[ii] == 'tanh':
                t = self.tanh(self.layers[ii](t))
            elif self.activations[ii] == 'relu':
                t = self.relu(self.layers[ii](t))
            else:
                t = self.layers[ii](t)
        y_hat = torch.matmul(A, t)
        sigma = pyro.sample("sigma", dist.Uniform(0, 0.01))  # Example shape and rate parameters
        
        
        if y != None:
            with pyro.plate("data", len(y)):
                obs = pyro.sample("obs", dist.Normal(y_hat[:,0], sigma), obs=y)

        return t.view(-1)
    

def generate_bnn_realization_plot(bnn, t, A, name, problem):
    # Generate prior realizations, A not used
    realizations = np.empty((len(t), 10))
    for ii in range(0, 10):
        realizations[:,ii] = bnn.forward(t, A)
    plt.plot(t, realizations)
    plt.savefig(f'plots/realizations/{problem}_{name}_realization.png')



def prior(t):
    gaussian = torch.distributions.normal.Normal(loc=torch.tensor([0.0]), scale=torch.tensor([5.0]))
    weights = gaussian.sample(sample_shape=(100,100))
    biases = gaussian.sample(
                sample_shape=t.shape
            )
    
    weights = weights[:,:,0]

    x = torch.tanh(torch.matmul(weights,t.view(-1, 1)) + biases)

    gaussian = torch.distributions.normal.Normal(loc=torch.tensor([0.0]), scale=torch.tensor([1.0])/np.sqrt(100))
    weights = gaussian.sample(sample_shape=(100,100))
    biases = gaussian.sample(
                sample_shape=t.shape
            )
    
    weights = weights[:,:,0]

    x = torch.matmul(weights,x.view(-1, 1)) + biases
#    x =  np.tanh(x@weights2 + bias2)
#    x =  x@weights3 + bias3
    return x.view(-1)





if __name__ == '__main__':
    
    # Parse the config argument
    parser = argparse.ArgumentParser(description="1D-deconvolution solving with BNN prior")
    parser.add_argument('--file', type=str, required=True, help='config file to use')
    parser.add_argument('--config', type=str, required=True, help='config to use')
    args = parser.parse_args()
    config = yaml.safe_load(open(f"codes/config/{args.file}"))[args.config]
    
    # Define if trained on cpu or gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = 'cpu'
    torch.set_default_device(device)

    # Get the initial parameters from the config file
    n_t = config['n_t']
    n_y = config['n_y']
    domain = config['domain']
    sigma_noise = config['sigma_noise']
    problem_type = config['problem']


    t = torch.linspace(-1, 1, n_t)
    A = np.eye(n_t)
    bnn_model = BNN(n_in=n_t,
                    n_out=n_t,
                    layers=config['bnn']['layers'])
    generate_bnn_realization_plot(bnn_model, t, A, args.config, problem)
    
