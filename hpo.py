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
experiment.config.trial_command = 'python model.py'
experiment.config.trial_code_directory = '.'

# Configuring the search space
experiment.config.search_space = search_space

experiment.config.tuner.name = 'TPE'

experiment.config.tuner.class_args = {
    'optimize_mode': 'maximize',
    'seed': 9999,
    'tpe_args': {
        'constant_liar_type': 'mean',
        'n_startup_jobs': 10,
        'n_ei_candidates': 20,
        'linear_forgetting': 10,
        'prior_weight': 0,
        'gamma': 0.6
    }
}

experiment.config.debug = True

# Setting a name for the experiment
experiment.config.experiment_name = f'TPE - Detailed config and tuner version'

# Setting up number of trials to run -> Sets of hyperparameters and trial concurrency
experiment.config.max_trial_number = 50  # Change to a higher number -> 50
experiment.config.trial_concurrency = 10

# Running the experiment on portal
experiment.run(8059)

# Stopping the experiment
experiment.stop()
