import wandb
import subprocess
import itertools
import os

# Define the sweep configuration
sweep_config = {
    'method': 'random',
    'parameters': {
        'train_method': {
            'values': ['esd-u']
        },
        'concept_pair': {
            'values': [
                'cowboy-cowboy hat', 'doctor-man'
            ]
        }
    }
}

#  'values': [
#                 'a car-a truck', 'a city-a village', 'a cat-a dog', 'a van gogh painting-a vermeer painting', 'a horse-a zebra', 'a church-a temple', 'a tiger-a lion'
#             ]

# Save sweep config to a file for reproducibility
import yaml
with open("sweep_config.yaml", "w") as f:
    yaml.dump(sweep_config, f)

import random

param_names = list(sweep_config['parameters'].keys())
param_values = [sweep_config['parameters'][k]['values'] for k in param_names]

num_runs = 10  # Set how many random runs you want

print(f"Total random runs: {num_runs}")

run_counter = 0
for i in range(num_runs):
    # Randomly sample a value for each parameter
    params = {name: random.choice(values) for name, values in zip(param_names, param_values)}
    print(f"\nLaunching random run {i+1}/{num_runs} with params: {params}")

    for lr in [5e-5]:
        

        for use_likelihood_ratio in [0,1]:

            if use_likelihood_ratio == 1:
                for ratio_scale in [100.0]:
                    for beta_1 in [0.5]:
                        for beta_2 in [0.9]:
            
                            # Set environment variables for wandb run grouping
                            os.environ["WANDB_RUN_GROUP"] = "manual_sweep"
                            os.environ["WANDB_NAME"] = f"sweep_run_{run_counter}_use_likelihood_ratio_{use_likelihood_ratio}"

                            # Build the command to run train.py with these params plus use_ratio_scale
                            cmd = [
                                "python", "train_my_from.py",
                                "--concept_pair", str(params['concept_pair']),
                                "--train_method", str(params['train_method']),
                                "--ratio_scale", str(ratio_scale),
                                "--lr", str(lr),
                                "--use_likelihood_ratio", str(use_likelihood_ratio),
                                "--beta_1", str(beta_1),
                                "--beta_2", str(beta_2),
                            ]

                            print("Running command:", " ".join(cmd))
                            subprocess.run(cmd)
                            run_counter += 1
                    
            else:
                # Set environment variables for wandb run grouping
                os.environ["WANDB_RUN_GROUP"] = "manual_sweep"
                os.environ["WANDB_NAME"] = f"sweep_run_{run_counter}_use_likelihood_ratio_{use_likelihood_ratio}"

                # Build the command to run train.py with these params plus use_ratio_scale
                cmd = [
                    "python", "train_my_from.py",
                    "--concept_pair", str(params['concept_pair']),
                    "--train_method", str(params['train_method']),
                    "--ratio_scale", str(1),
                    "--lr", str(lr),
                    "--use_likelihood_ratio", str(use_likelihood_ratio),
                ]

                print("Running command:", " ".join(cmd))
                subprocess.run(cmd)
                run_counter += 1
