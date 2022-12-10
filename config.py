# Copyright 2021 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Realize the parameter configuration function of dataset, model, training and verification code."""
import random
import os
import numpy as np
import torch
from torch.backends import cudnn

# Random seed to maintain reproducible results
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

# Use GPU for training by default
device = torch.device("cuda", 0)

# Turning on when the image size does not change during training can speed up training
cudnn.benchmark = True

# Image magnification factor
upscale_factor = 4

# Current configuration parameter method -> Change to test for testing
mode = "train"

# Experiment name, easy to save weights and log files
exp_name = "vdsr_baseline"

if mode == "train":
    # Dataset
    # train_image_dir = os.path.join('data', 'DIV2K', 'train')
    # valid_image_dir = os.path.join('data', 'DIV2K', 'valid')
    train_image_dir = '/ocean/projects/cis220070p/jshah2/div2k/train_hpo'
    valid_image_dir = '/ocean/projects/cis220070p/jshah2/div2k/valid_hpo'
    test_image_dir = '/ocean/projects/cis220070p/jshah2/Set5/GTmod12'
    # test_image_dir = os.path.join('data', 'Set5')

    image_size = 41
    # batch_size = 16
    num_workers = 4

    # Incremental training and migration training
    start_epoch = 0
    resume = ""

    # Total num epochs
    epochs = 10

    # SGD optimizer parameter
    # model_lr = 0.1
    # model_momentum = 0.9
    model_weight_decay = 1e-4
    model_nesterov = False

    # StepLR scheduler parameter
    lr_scheduler_step_size = epochs // 4
    lr_scheduler_gamma = 0.1

    # gradient clipping constant
    clip_gradient = 0.01

    print_frequency = 200

if mode == "valid":
    # Test data address
    sr_dir = os.path.join('results', 'test', '{}'.format(exp_name))
    hr_dir = os.path.join('data', 'Set5')

    model_path = os.path.join('results', '{}'.format(exp_name), 'best.pth.tar')
