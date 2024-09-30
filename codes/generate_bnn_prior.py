import pickle
import argparse
import yaml
import random

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
from bnn import BNN

'''
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "mathptmx",
    'font.size': 20
})
'''

def generate_realization_plot(configs):
    # Get the initial parameters from the config file
    fig, axs = plt.subplots(1, 3, figsize=(16, 5))
    results = dict()
    for ii, config in enumerate(configs):
        config = configs[config]
        n_t = config['n_t']
        domain = config['domain']

        seed = config['training_parameters']['random_seed']
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


        t = torch.linspace(-1, 1, n_t)
        A = torch.eye(n_t)
        bnn_model = BNN(n_in=1,
                        n_out=1,
                        layers=config['bnn']['layers'])
        realizations = generate_bnn_realization(bnn_model, t, A)
        results[str(ii)] = realizations
        axs[ii].plot(t, realizations)
    
    with open(f'results/prior/priors.pickle', 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    plt.savefig(f'plots/prior/realizations.eps', format='eps')
    

def generate_bnn_realization(bnn, t, A):
    # Generate prior realizations, A not used
    realizations = np.empty((len(t), 10))
    for ii in range(0, 10):
        realizations[:,ii] = bnn.forward(t, A)
    return realizations



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
    args = parser.parse_args()
    configs = yaml.safe_load(open(f"codes/config/{args.file}"))
    
    # Define if trained on cpu or gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = 'cpu'
    torch.set_default_device(device)

    generate_realization_plot(configs)
    
