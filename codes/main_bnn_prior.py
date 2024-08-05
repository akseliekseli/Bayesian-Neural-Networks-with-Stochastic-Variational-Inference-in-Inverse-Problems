import pickle
import argparse

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn

import yaml

import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoDiagonalNormal, AutoNormal
from pyro.optim import Adam

from tqdm.auto import trange

from models import deconvolution



#plt.rcParams["font.family"] = "Deja Vu"
plt.rcParams['font.size'] = 32


def problem_system(grid: np.array, params)-> np.array:

    output = np.zeros(grid.shape)
    for idx, point in enumerate(grid):
        if point <= -0.6:
            output[idx] = params[0]
        elif point <= 0.0:
            output[idx] = params[1]+ 0.2*np.sin(1.7*np.pi*point*2)
        elif point <= 0.4:
            output[idx] = params[2]
        elif point <= 0.6:
            #output[idx] = params[3]
            output[idx] = params[3] #+ 0.3*np.sin(2*np.pi*point*2)
        else:
            output[idx] = point*params[4]
    
    return output


class BNN(PyroModule):

    def __init__(self, n_in, n_out, n_layers, layers):
        super().__init__()

        self.n_layers = n_layers

        self.layers = PyroModule[torch.nn.ModuleList]([
                                PyroModule[nn.Linear](n_in, n_out)
        for j in range(n_layers)
        ])  

        for ii, layer in enumerate(layers):

            if layers[layer]['type'] == 'cauchy':
                self.layers[ii].weight = PyroSample(dist.Cauchy(0.,
                                                    torch.tensor(layers[layer]['weight'])).expand([n_out, n_out]).to_event(2))
                
                self.layers[ii].bias = PyroSample(dist.Cauchy(0.,
                                                torch.tensor(layers[layer]['bias'])).expand([n_out]).to_event(1))
            elif layers[layer]['type'] == 'gaussian':
                self.layers[ii].weight = PyroSample(dist.Normal(0.,
                                                    torch.tensor(layers[layer]['weight'])).expand([n_out, n_out]).to_event(2))
                
                self.layers[ii].bias = PyroSample(dist.Normal(0.,
                                                torch.tensor(layers[layer]['bias'])).expand([n_out]).to_event(1))

            else:
                print('Invalid layer!')

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x, A, y=None):
        
        x = x#.reshape(-1, 1)

        mu = self.relu(self.layers[0](x))
        for ii in range(1, self.n_layers):
            mu = self.relu(self.layers[ii](mu))

        
        y_hat = torch.matmul(A, mu)
        
        sigma = pyro.sample("sigma", dist.Uniform(0.,
                                                torch.tensor(0.01)))
        with pyro.plate("data", n_y):
            obs = pyro.sample("obs", dist.Normal(y_hat, sigma), obs=y)
        
        return mu
    

def generate_the_problem(n_x: int,
                         n_y: int,
                         domain: list,
                         sigma_noise: float):
    # Generate the grid
    t = np.linspace(domain[0],domain[1], n_y)
    t = np.round(t, 3)
    h = domain[1] / n_y
    # Create the convolution matrix A
    model = deconvolution(int(np.round(n_x/2)), int(n_x/16), 'reflect')
    A = model.linear_operator(n_x)
    #A = A[1::2, :]
    #A[0,0] = 0
    #A[-1, -1] = 0

    # Parameters used for the problem
    problem_params = [0, 0.8, 0., 1.0, 0]

    # Generate grid points
    x = np.linspace(domain[0], domain[1] - h, n_x)
    # Construct the true function
    true = problem_system(x, problem_params)
    temp = A@true
    ind = temp > 0
    temp *= ind

    # Create y_data with noise
    y_data = temp + np.random.normal(0, sigma_noise, true.shape)

    return t, x, A, true, y_data


def training_bnn_gpu(config, t, A, y_data):
    # Set Pyro random seed
    pyro.set_rng_seed(config['training_parameters']['random_seed'])

    # Define Pyro BNN object and training parameters
    bnn_model = BNN(n_in=n_y, n_out=n_x, n_layers=config['bnn']['n_layers'])
    guide = AutoDiagonalNormal(bnn_model)
    adam_params = {"lr": config['training_parameters']['learning_rate'],
                "betas": (0.9, 0.999)}
    optimizer = Adam(adam_params)
    svi = pyro.infer.SVI(bnn_model, guide, optimizer, loss=Trace_ELBO())
    num_iterations = config['training_parameters']['svi_num_iterations']
    progress_bar = trange(num_iterations)

    t_gpu = torch.from_numpy(t).float().cuda()
    y_gpu = torch.from_numpy(y_data).float().cuda()
    A_gpu = torch.from_numpy(A).float().cuda()

    for j in progress_bar:
        loss = svi.step(t_gpu, A_gpu, y_gpu)
        progress_bar.set_description("[iteration %04d] loss: %.4f" % (j + 1, loss / len(t_gpu)))

    # Get predictions for the solution
    predictive = pyro.infer.Predictive(bnn_model, guide=guide, num_samples=2000, return_sites=["_RETURN"])
    preds_gpu = predictive(t_gpu, A_gpu)
    x_preds_cpu = preds_gpu['_RETURN'].cpu()
    
    return x_preds_cpu


def generate_the_problem(n_x: int,
                         n_y: int,
                         domain: list,
                         sigma_noise: float):
    # Generate the grid
    t = np.linspace(domain[0],domain[1], n_y)
    t = np.round(t, 3)
    h = domain[1] / n_y
    # Create the convolution matrix A
    model = deconvolution(int(np.round(n_x/2)), int(n_x/16), 'reflect')
    A = model.linear_operator(n_x)
    #A = A[1::2, :]
    #A[0,0] = 0
    #A[-1, -1] = 0

    # Parameters used for the problem
    problem_params = [0, 0.8, 0., 1.0, 0]

    # Generate grid points
    x = np.linspace(domain[0], domain[1] - h, n_x)
    # Construct the true function
    true = problem_system(x, problem_params)
    temp = A@true
    ind = temp > 0
    temp *= ind

    # Create y_data with noise
    y_data = temp + np.random.normal(0, sigma_noise, true.shape)

    return t, x, A, true, y_data

def training_bnn_cpu(config, t, A, y_data):
    # Set Pyro random seed
    pyro.set_rng_seed(config['training_parameters']['random_seed'])

    t= torch.from_numpy(t).float()
    y_data = torch.from_numpy(y_data).float()
    A = torch.from_numpy(A).float()

    # Define Pyro BNN object and training parameters
    bnn_model = BNN(n_in=n_y,
                    n_out=n_x,
                    n_layers=config['bnn']['n_layers'],
                    layers=config['bnn']['layers'])
    guide = AutoDiagonalNormal(bnn_model)
    adam_params = {"lr": config['training_parameters']['learning_rate'],
                "betas": (0.9, 0.999)}
    optimizer = Adam(adam_params)
    svi = pyro.infer.SVI(bnn_model, guide, optimizer, loss=Trace_ELBO())
    num_iterations = config['training_parameters']['svi_num_iterations']
    progress_bar = trange(num_iterations)

    for j in progress_bar:
        loss = svi.step(t, A, y_data)
        progress_bar.set_description("[iteration %04d] loss: %.4f" % (j + 1, loss / len(t)))

    # Get predictions for the solution
    predictive = pyro.infer.Predictive(bnn_model, guide=guide, num_samples=2000, return_sites=["_RETURN"])
    preds = predictive(t, A)
    x_preds = preds['_RETURN']
    
    return x_preds


def calculate_mean_and_quantile(x_preds):
    x_mean = torch.mean(x_preds, axis=0)
    lower_quantile = torch.quantile(x_preds, 0.05, axis=0)
    upper_quantile = torch.quantile(x_preds, 0.95, axis=0)

    return x_mean, lower_quantile, upper_quantile


def plot_results(config, t, x, true, y_data, x_preds):
    x_mean, lower_quantile, upper_quantile = calculate_mean_and_quantile(x_preds)

    plt.figure()
    plt.plot(x, true, label='true')
    plt.plot(t, y_data, label='data')

    line, = plt.plot(t, x_mean, label='inverse solution')
    # Plot the quantile range as a shaded area
    plt.fill_between(x, lower_quantile, upper_quantile, color=line.get_color(), alpha=0.5, label='90% quantile range')
    #plt.plot(t, A@x_solution.numpy(), label='A @ solution')
    plt.axis([domain[0], domain[1], -0.1, 1.5])
    plt.xlabel('t')
    plt.ylabel('x')

    plt.savefig(f"plots/{config['name']}_solution.png")


def plot_problem(config, t, x, true, y_data):
    plt.figure()
    plt.plot(x, true, label='true')
    plt.plot(t, y_data, label='data')

    plt.axis([domain[0], domain[1], -0.5, 1.5])
    plt.savefig(f"plots/{config['name']}_problem.png")



if __name__ == '__main__':
    # Parse the config argument
    parser = argparse.ArgumentParser(description="1D-deconvolution solving with BNN prior")
    parser.add_argument('--type', type=str, required=True, help='Type of config')
    parser.add_argument('--config', type=str, required=True, help='Config to use (use config file)')
    args = parser.parse_args()
    config = yaml.safe_load(open("codes/config/config.yml"))[args.type][args.config]
    
    # Define if trained on cpu or gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = 'cpu'
    torch.set_default_device(device)

    # Get the initial parameters from the config file
    n_x = config['n_x']
    n_y = config['n_y']
    domain = config['domain']
    sigma_noise = config['sigma_noise']
    
    t, x, A, true, y_data = generate_the_problem(n_x, n_y, domain, sigma_noise)

    
    # Convert data to PyTorch tensors
    if device != 'cpu':
        x_preds = training_bnn_gpu(config, t, A, y_data)
    else:
        x_preds = training_bnn_cpu(config, t, A, y_data)
    
    
    results = dict({'t': t,
                            'x': x,
                            'true': true,
                            'y_data': y_data,
                            'x_preds': x_preds})
    
    with open(f'results/{config['name']}.pickle', 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)


    plot_problem(config, t, x, true, y_data)
    plot_results(config, t, x, true, y_data, x_preds)
    

    
        