import numpy as np

import matplotlib.pyplot as plt
import scipy as sp
import scipy.stats as stats


import torch
import torch.nn as nn
import torch.optim as optim

import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample


from models import deconvolution


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

device = 'cpu'

torch.set_default_device(device)

print(torch.get_default_device())


# Parameters
T = 0.05
n_y = 100

domain = [-1, 1]
sigma_noise = 0.01**2

n_x = n_y


t = np.linspace(domain[0],domain[1], n_y)
t = np.round(t, 3)
d_k = 40

h = domain[1] / n_y

n_datasize = 6000
dataset = np.zeros((n_datasize, n_x))
x = np.linspace(domain[0],domain[1], n_x)

for ii in range(0, n_datasize):
    n_params = np.random.randint(4, 6)
    idxs = np.sort(np.random.randint(0+5, n_x-5, n_params))
    params = np.sort(np.random.uniform(0, 2, n_params))
    params[::2] = 0
    true = np.zeros(x.shape)

    for jj in range(0, len(idxs[1:])):
        true[idxs[jj-1]:idxs[jj]] = params[jj]

    dataset[ii, :] = true



model = deconvolution(int(np.round(n_x/2)), int(n_x/16), 'reflect')
A = model.linear_operator(n_x)
#A = A[1::2, :]

y_data = np.zeros((n_datasize, n_y))

for ii in range(0, n_datasize):
    f = dataset[ii, :]
    f = A@f
    ind = f > 0
    f *= ind

    # Create y_data with noise
    y_data[ii, :] = f + np.random.normal(0, sigma_noise, f.shape)



#Cauchy-jakaumia BNN:ään. 

class BNN(PyroModule):

    def __init__(self, h1, h2):
        super().__init__()
        self.fc1 = PyroModule[nn.Linear](h1, h2)
        self.fc1.weight = PyroSample(dist.Normal(0.,
                                                torch.tensor(0.05)).expand([h2, h1]).to_event(2))
        self.fc1.bias = PyroSample(dist.Normal(0.,
                                               torch.tensor(0.05)).expand([h2]).to_event(1))
        
        #self.fc1 = nn.Linear(h1, h2)

        self.fc2 = PyroModule[nn.Linear](h1, h2)
        self.fc2.weight = PyroSample(dist.Cauchy(0.,
                                                torch.tensor(3.0)).expand([h2, h2]).to_event(2))
        self.fc2.bias = PyroSample(dist.Cauchy(0.,
                                               torch.tensor(3.0)).expand([h2]).to_event(1))

        self.fc3 = PyroModule[nn.Linear](h1, h2)
        self.fc3.weight = PyroSample(dist.Normal(0.,
                                                torch.tensor(0.05)).expand([h2, h2]).to_event(2))
        self.fc3.bias = PyroSample(dist.Normal(0.,
                                               torch.tensor(0.05)).expand([h2]).to_event(1))

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x, y=None):
        
        x = x#.reshape(-1, 1)

        mu = self.tanh(self.fc1(x))
        mu = self.relu(self.fc2(mu))
        mu = self.relu(self.fc3(mu))
        #mu = x
        sigma = pyro.sample("sigma", dist.Uniform(0.,
                                                torch.tensor(0.05)))
    
        with pyro.plate("data", n_x):
            obs = pyro.sample("obs", dist.Normal(mu, sigma), obs=y)
        
        return mu
    

bnn_model = BNN(h1=n_y, h2=n_x)

# Set Pyro random seed
pyro.set_rng_seed(42)

nuts_kernel = pyro.infer.NUTS(bnn_model, jit_compile=True)

# Define MCMC sampler, get 50 posterior samples
bnn_mcmc = pyro.infer.MCMC(nuts_kernel,
                        num_samples=50,
                        warmup_steps=50)


# Convert data to PyTorch tensors
x_train = torch.from_numpy(y_data).float()
y_train = torch.from_numpy(dataset).float()

x_test = x_train[-1,:]
y_test = y_train[-1,:]

x_train = x_train[:-2,:]
y_train = y_train[:-2,:]

# Run MCMC
#bnn_mcmc.run(x_train_gpu, y_train_gpu)

from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoDiagonalNormal, AutoNormal
from tqdm.auto import trange
from pyro.optim import Adam

guide = AutoDiagonalNormal(bnn_model)
adam_params = {"lr": 0.0005, "betas": (0.90, 0.999)}
optimizer = Adam(adam_params)
svi = pyro.infer.SVI(bnn_model, guide, optimizer, loss=Trace_ELBO())

num_iterations = 10
progress_bar = trange(num_iterations)

for j in progress_bar:

    for ii in range(0, x_train.shape[0]):
        loss = svi.step(x_train[ii,:], y_train[ii,:])
    progress_bar.set_description("[iteration %04d] loss: %.4f" % (j + 1, loss / len(x_train)))


#predictive = pyro.infer.Predictive(model=bnn_model, posterior_samples=bnn_mcmc.get_samples())
predictive = pyro.infer.Predictive(bnn_model, guide=guide, num_samples=5000)

#x_train_gpu = torch.from_numpy(y_data).float().cuda()[1,:]
#y_train_gpu = torch.from_numpy(dataset).float().cuda()[1,:]

preds = predictive(x_test)


# Save model and guide parameters
torch.save(predictive, 'bnn_prior_cpu.pt')

preds = preds['obs']

plt.plot(x, torch.mean(preds, axis=0))
plt.plot(t, x_test)
plt.plot(x, y_test)


plt.legend(['BNN', 'Noisy', 'True'])

plt.savefig('bnn_prior_cpu.png')


