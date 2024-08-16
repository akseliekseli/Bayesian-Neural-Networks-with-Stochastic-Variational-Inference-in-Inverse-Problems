The main code is in file .codes/main_bnn_prior.py. It utilizes Pyro library for the BNN and SVI. 

To run this, you need to use the config file in config folder to provide the parameters. 
You can run the code with command ```python3 codes/main_bnn_prior.py --problem_type initial --experiment_type initial --config continuous```

There is also an example on prior generation in file .codes/generate_bnn_prior.py
This doesn't use Pyro but instead Numpy arrays for the prior generation.
