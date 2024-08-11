import yaml
import os
import argparse

config = yaml.safe_load(open("codes/config/config.yml"))

# Parse the config argument
parser = argparse.ArgumentParser(description="1D-deconvolution solving with BNN prior")
parser.add_argument('--experiment', type=str, required=True, help='Type of config, all to run all configs')
args = parser.parse_args()

experiment = args.experiment

if experiment == 'all':
    for problem_type in config:
        for exp_type in config[problem_type]:
            for conf in config[problem_type][exp_type]:
                print(f'RUNNING CONFIG: {conf}\n')
                os.system(f"python3 codes/main_bnn_prior.py --problem_type {problem_type} --experiment_type {exp_type} --config {conf}")
else:
    for problem_type in config:
        if problem_type == 'initial':
            pass
        else:
            for conf in config[problem_type][experiment]:
                print(f'RUNNING CONFIG: {conf}\n')
                os.system(f"python3 codes/main_bnn_prior.py --problem_type {problem_type} --experiment_type {experiment} --config {conf}")