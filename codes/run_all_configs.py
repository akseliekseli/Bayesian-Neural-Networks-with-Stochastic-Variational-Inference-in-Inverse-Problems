import yaml
import os
import argparse

config = yaml.safe_load(open("codes/config/config.yml"))

# Parse the config argument
parser = argparse.ArgumentParser(description="1D-deconvolution solving with BNN prior")
parser.add_argument('--conf_family', type=str, required=True, help='Type of config, all to run all configs')
args = parser.parse_args()

conf_family = args.conf_family

if conf_family == 'all':
    for type in config:
        for conf in config[conf_family]:
            print(f'RUNNING CONFIG: {conf}\n')
            os.system(f"python3 codes/main_bnn_prior.py --type {conf_family} --config {conf}")
else:
    for conf in config[conf_family]:
        print(f'RUNNING CONFIG: {conf}\n')
        os.system(f"python3 codes/main_bnn_prior.py --type {conf_family} --config {conf}")