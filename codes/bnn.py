import numpy as np
import torch
import torch.nn as nn

import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample


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
                    self.layers[ii].weight = PyroSample(dist.Cauchy(0., torch.tensor(weight)).expand([self.layers[ii].out_features, n_in]).to_event(2))
                elif ii == self.n_layers-1:
                    self.layers[ii].weight = PyroSample(dist.Cauchy(0., torch.tensor(weight)).expand([n_out, self.layers[ii].in_features]).to_event(2))
                else:
                    self.layers[ii].weight = PyroSample(dist.Cauchy(0., torch.tensor(weight)).expand([self.layers[ii].out_features, self.layers[ii].in_features]).to_event(2))

                if ii == self.n_layers - 1:  # Special case for the last layer
                    self.layers[ii].bias = PyroSample(dist.Cauchy(0., torch.tensor(bias)).expand([1, 1]).to_event(2))
                    #pass
                else:
                    self.layers[ii].bias = PyroSample(dist.Cauchy(0., torch.tensor(bias)).expand([1, self.layers[ii].out_features]).to_event(2))
                
            elif layers[layer]['type'] == 'gaussian':
                if ii == 0:  # First layer
                    self.layers[ii].weight = PyroSample(dist.Normal(0., torch.tensor(weight)).expand([self.layers[ii].out_features, n_in]).to_event(2))
                elif ii == self.n_layers-1:
                    self.layers[ii].weight = PyroSample(dist.Normal(0., torch.tensor(weight)).expand([n_out, self.layers[ii].in_features]).to_event(2))
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
        sigma = pyro.sample("sigma", dist.Uniform(0, 0.01))
        
        
        if y != None:
            with pyro.plate("data", len(y)):
                obs = pyro.sample("obs", dist.Normal(y_hat[:,0], sigma), obs=y)

        return t.view(-1)