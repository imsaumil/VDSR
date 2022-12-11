# Importing required libraries
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

# Defining the HPO tuner algorithm to be used
experiment.config.tuner.name = 'Evolution'
experiment.config.tuner.class_args = {
        'optimize_mode': 'maximize',
        'population_size': 125
}

experiment.config.debug = True

# Setting a name for the experiment
experiment.config.experiment_name = f'VDSR HPO'

# Setting up number of trials to run -> Sets of hyperparameters and trial concurrency
experiment.config.max_trial_number = 50  # Change to a higher number -> 50
experiment.config.trial_concurrency = 5

# Running the experiment on portal
experiment.run(8059)

# Stopping the experiment
experiment.stop()
