import yaml
import os

config = yaml.safe_load(open("codes/config/config.yml"))

for config_name in config:
    print(f'RUNNING CONFIG: {config_name}\n')
    os.system(f"python3 codes/main_bnn_prior.py --config {config_name}")