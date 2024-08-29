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



#plt.rcParams["font.family"] = "Deja Vu"
#plt.rcParams['font.size'] = 32

def problem_system_combined(grid: np.array)-> np.array:
    params = [0, 0.7, 0., 1.0, 0]
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

def problem_system_discrete(grid: np.array)-> np.array:

    
    params = [0., 0.8, 0., -0.7, 0.0, 0.8, 0.]

    output = np.zeros(grid.shape)
    for idx, point in enumerate(grid):
        if point <= -0.6:
            output[idx] = params[0]
        elif point <= -0.5:
            output[idx] = params[1]
        elif point <= 0.0:
            output[idx] = params[2]
        elif point <= 0.2:
            output[idx] = params[3]
        elif point <= 0.5:
            output[idx] = params[4]
        elif point <= 0.6:
            output[idx] = params[5]
        else:
            output[idx] = params[6]
    
    return output


def problem_system_continuous(grid: np.array)-> np.array:

    # Define boundary and internal points
    x = np.array([-1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.3, -0.1, 0.0, 0.2, 0.5, 0.6, 1])
    y = np.array([0, 0.0, 0.0, -0.6, 0., 0.6, 0.2,  -0.1, 0.4, 0.8, 0.1, -0.1, -0.2])

    # Create a cubic spline interpolation of the points
    cs = CubicSpline(x, y)
    output = cs(grid)
    
    return output


class BNNGuide(PyroModule):
    def __init__(self, model):
        super().__init__()
        self.n_layers = model.n_layers
        self.layers = PyroModule[torch.nn.ModuleList]([])
        self.model = model
        for ii, layer in enumerate(model.layers):
            # Mean and std of weights
            self.layers.append(PyroModule[nn.Linear](layer.in_features, layer.out_features))
            
            if isinstance(layer.weight, PyroSample):
                weight_loc = pyro.param(f"weight_loc_{ii}", torch.randn_like(layer.weight.mean))
                weight_scale = pyro.param(f"weight_scale_{ii}", torch.randn_like(layer.weight.mean).abs())
                
                self.layers[ii].weight = PyroSample(dist.Normal(weight_loc, weight_scale).to_event(2))
            
            if isinstance(layer.bias, PyroSample):
                bias_loc = pyro.param(f"bias_loc_{ii}", torch.randn_like(layer.bias.mean))
                bias_scale = pyro.param(f"bias_scale_{ii}", torch.randn_like(layer.bias.mean).abs())
                
                self.layers[ii].bias = PyroSample(dist.Normal(bias_loc, bias_scale).to_event(2))
        
        self.sigma = pyro.param("sigma", torch.tensor(0.1), constraint=dist.constraints.positive)


    def forward(self, t, A, y=None):
        # Sample sigma
        pyro.sample("sigma", dist.Gamma(self.sigma_concentration, self.sigma_rate))

        # Passing the input through the network
        for ii in range(self.n_layers):
            t = self.layers[ii](t)
            if self.model.activations[ii] == 'tanh':
                t = torch.tanh(t)
            elif self.model.activations[ii] == 'relu':
                t = torch.relu(t)
        
        return t.view(-1)



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

 
def generate_bnn_realization_plot(bnn, t, A):
    # Generate prior realizations, A not used
    realizations = np.empty((len(t), 10))
    for ii in range(0, 10):
        realizations[:,ii] = bnn.forward(t, A)
    plt.plot(t, realizations)
    plt.savefig('realization.png')


def generate_the_problem(problem_type: str,
                        n_t: int,
                        n_y: int,
                        domain: list,
                        sigma_noise: float):
    # Generate the grid
    t = np.linspace(domain[0],domain[1], n_t)
    t = np.round(t, 3)
    h = domain[1] / n_t
    # Create the convolution matrix A
    model = deconvolution(int(np.round(n_y/8)), int(n_y/16), 'reflect')
    A = model.linear_operator(n_y)
    #A = A[1::2, :]
    #A[0,0] = 0
    #A[-1, -1] = 0


    # Generate grid points
    x = np.linspace(domain[0], domain[1] - h, n_y)
    # Construct the true function
    if problem_type == 'discrete':
        true = problem_system_discrete(x)
    if problem_type == 'continuous':
        true = problem_system_continuous(x)
    temp = A@true
    #ind = temp > 0
    #temp *= ind

    # Create y_data with noise
    y_data = temp + np.random.normal(0, sigma_noise, true.shape)

    return t, x, A, true, y_data


def training_bnn_gpu(config, t, A, y_data):
    # Set Pyro random seed
    pyro.set_rng_seed(config['training_parameters']['random_seed'])

    # Define Pyro BNN object and training parameters
    bnn_model = BNN(n_in=n_y,
                    n_out=n_t,
                    layers=config['bnn']['layers'])
    guide = AutoDiagonalNormal(bnn_model)
    adam_params = {"lr": config['training_parameters']['learning_rate'],
                "betas": (0.9, 0.999)}
    optimizer = Adam(adam_params)
    guide = BNNGuide(bnn_model)
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
    predictive = pyro.infer.Predictive(bnn_model, guide=guide, num_samples=20000, return_sites=["_RETURN"])
    preds_gpu = predictive(t_gpu, A_gpu)
    x_preds_cpu = preds_gpu['_RETURN'].cpu()
    
    return x_preds_cpu


def training_bnn_cpu(config, t, A, y_data):
    # Set Pyro random seed
    pyro.set_rng_seed(config['training_parameters']['random_seed'])

    t= torch.from_numpy(t).float()
    y_data = torch.from_numpy(y_data).float()
    A = torch.from_numpy(A).float()

    # Define Pyro BNN object and training parameters
    bnn_model = BNN(n_in=n_y,
                    n_out=n_t,
                    layers=config['bnn']['layers'])
    guide = AutoDiagonalNormal(bnn_model)
    adam_params = {"lr": config['training_parameters']['learning_rate'],
                "betas": (0.9, 0.999)}
    optimizer = Adam(adam_params)
    svi = pyro.infer.SVI(bnn_model, guide, optimizer, loss=Trace_ELBO())
    num_iterations = config['training_parameters']['svi_num_iterations']
    progress_bar = trange(num_iterations)
    generate_bnn_realization_plot(bnn_model, t, A)
    for j in progress_bar:
        loss = svi.step(t, A, y_data)
        progress_bar.set_description("[iteration %04d] loss: %.4f" % (j + 1, loss / len(t)))

    # Get predictions for the solution
    predictive = pyro.infer.Predictive(bnn_model, guide=guide, num_samples=2000, return_sites=["_RETURN", "sigma"])
    preds = predictive(t, A)
    x_preds = preds['_RETURN']
    print(f'sigma: {preds["sigma"]}')
    return x_preds


def calculate_mean_and_quantile(x_preds):
    x_mean = torch.mean(x_preds, axis=0)
    lower_quantile = torch.quantile(x_preds, 0.05, axis=0)
    upper_quantile = torch.quantile(x_preds, 0.95, axis=0)

    return x_mean, lower_quantile, upper_quantile


def plot_results(config, t, x, true, y_data, x_preds):
    x_mean, lower_quantile, upper_quantile = calculate_mean_and_quantile(x_preds)

    print(f'X {x_preds.shape}')

    plt.figure()
    plt.plot(x, true, label='true')
    plt.plot(t, y_data, label='data')
    line, = plt.plot(t, x_mean, label='inverse solution')
    # Plot the quantile range as a shaded area
    plt.fill_between(x, lower_quantile, upper_quantile, color=line.get_color(), alpha=0.5, label='90% quantile range')
    #plt.plot(t, x_preds[0:3, :].T)
    #plt.plot(t, A@x_mean.numpy(), label='A @ solution')
    plt.axis([domain[0], domain[1], -1.0, 1.7])
    plt.xlabel('t')
    plt.ylabel('x')

    plt.savefig(f"plots/{config['name']}_solution.png")


def plot_problem(config, t, x, true, y_data):
    plt.figure()
    plt.plot(x, true, label='true')
    plt.plot(t, y_data, label='data')

    plt.axis([domain[0], domain[1], -1.0, 1.7])
    plt.savefig(f"plots/{config['name']}_problem.png")



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

    np.random.seed(config['training_parameters']['random_seed'])
    
    t, x, A, true, y_data = generate_the_problem(problem_type,
                                                n_t,
                                                n_y,
                                                domain,
                                                sigma_noise)

    
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
    

    
        