# Importing required libraries
import sys
import json
from nni.experiment import Experiment

with open('./search_space.json', 'r') as file:
    data = file.read()

# Importing the custom search spaces
search_space = json.loads(data)

# Setting up the NNI experiment on local machine
experiment = Experiment('local')

# Conducting NNI evaluation in trail mode
experiment.config.trial_command = 'python3 model.py'
experiment.config.trial_code_directory = '.'

# Configuring the search space
experiment.config.search_space = search_space

# This tuner needs additional install -> pip install nni[BOHB]
# experiment.config.tuner.name = 'BOHB'
experiment.config.tuner.name = 'TPE'

# experiment.config.tuner.class_args = {
#     'optimize_mode': 'maximize',
#     'min_budget': 1,
#     'max_budget': 600,
#     'eta': 3,
#     'min_points_in_model': 7,
#     'top_n_percent': 15,
#     'num_samples': 64,
#     'random_fraction': 0.33,
#     'bandwidth_factor': 3.0,
#     'min_bandwidth': 0.001
# }
# experiment.config.tuner.name = 'Hyperband'
# experiment.config.tuner.class_args = {
#     'optimize_mode': 'maximize',
#     'R': 60,
#     'eta': 3
# }

experiment.config.debug = True

# Setting a name for the experiment
experiment.config.experiment_name = f'TPE - Detailed config and tuner version'

# Setting up number of trials to run -> Sets of hyperparameters and trial concurrency
experiment.config.max_trial_number = 50  # Change to a higher number -> 50
experiment.config.trial_concurrency = 5

# Running the experiment on portal
experiment.run(8059)

# Stopping the experiment
experiment.stop()
