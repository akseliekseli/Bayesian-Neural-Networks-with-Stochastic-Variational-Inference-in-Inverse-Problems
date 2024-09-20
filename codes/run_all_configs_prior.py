import yaml
import os
import argparse

# Parse the config argument
parser = argparse.ArgumentParser(description="1D-deconvolution solving with BNN prior")
parser.add_argument('--file', type=str, required=True, help='config file to run fully, all to run all configs')
args = parser.parse_args()

file = args.file

if file == 'all':
    for filename in os.listdir("codes/config/"):
        config = yaml.safe_load(open(f"codes/config/{filename}"))
        for conf in config:
            print(f'RUNNING CONFIG: {conf}\n')
            os.system(f"python3 codes/generate_bnn_prior.py --file {filename} --config {conf}")
else:
    config = yaml.safe_load(open(f"codes/config/{args.file}"))
    for conf in config:
        print(f'RUNNING CONFIG: {conf}\n')
        os.system(f"python3 codes/generate_bnn_prior.py --file {file} --config {conf}")