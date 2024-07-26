import yaml
import os

config = yaml.safe_load(open("codes/config/config.yml"))

for type in config:
    for conf in config[type]:
        print(f'RUNNING CONFIG: {conf}\n')
        os.system(f"python3 codes/main_bnn_prior.py --type type --config {conf}")