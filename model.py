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
# ============================================================================
"""Realize the model definition function."""
from math import sqrt
import argparse
import nni
import torch
import torch.nn as nn
import config
from torchvision import datasets, transforms
from torchsummary import summary
from train import *

from dataset import CUDAPrefetcher
from dataset import TrainValidImageDataset, TestImageDataset


class ConvReLU(nn.Module):
    def __init__(self, channels: int) -> None:
        super(ConvReLU, self).__init__()
        self.conv = nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1), bias=False)
        self.relu = nn.ReLU(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out = self.relu(out)

        return out


class VDSR(nn.Module):
    def __init__(self) -> None:
        super(VDSR, self).__init__()
        # Input layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, (3, 3), (1, 1), (1, 1), bias=False),
            nn.ReLU(True),
        )

        # Features trunk blocks
        trunk = []
        for _ in range(18):
            trunk.append(ConvReLU(64))
        self.trunk = nn.Sequential(*trunk)

        # Output layer
        self.conv2 = nn.Conv2d(64, 1, (3, 3), (1, 1), (1, 1), bias=False)

        # Initialize model weights
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)

    # Support torch.script function
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.trunk(out)
        out = self.conv2(out)

        out = torch.add(out, identity)

        return out

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                module.weight.data.normal_(0.0, sqrt(
                    2 / (module.kernel_size[0] * module.kernel_size[1] * module.out_channels)))


if __name__ == '__main__':

    # Creating arguments parser
    parser = argparse.ArgumentParser(description='Super-resolution using VDSR')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()

    # Setting the random seed
    torch.manual_seed(args.seed)

    # Using GPU, if available
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # Device selected for training and testing
    device = torch.device("cuda" if args.cuda else "cpu")
    print("Device used: ", device, "\n")

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

        # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
        torch.backends.cudnn.allow_tf32 = True

        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    # Defining parameters to be tuned
    params = {
        'dropout_rate': 0.0,
        'lr': 0.001,
        'momentum': 0,
        "batch_size": 64
    }

    # Fetching the next optimized hyperparameter
    optimized_params = nni.get_next_parameter()
    params.update(optimized_params)
    print(params)

    # Using pre-fetcher to load data into memory
    train_prefetcher, valid_prefetcher, test_prefetcher = load_dataset(params['batch_size'])
    print("Load train dataset and valid dataset successfully.")

    # Creating the model
    model = VDSR().to(config.device)
    print("Build VDSR model successfully.")

    # Defining the loss function
    psnr_criterion, pixel_criterion = define_loss()
    print("Defined all loss functions successfully.")

    # Defining the optimizer to reduce the losses
    optimizer = define_optimizer(model, params['lr'], params['momentum'])
    print("Defined all optimizer functions successfully.")

    # Defining the optimizer scheduler
    scheduler = define_scheduler(optimizer)
    print("Defined all optimizer scheduler successfully.")

    # Using the pre-trained model data to get results efficiently using transfer learning
    print("Checking and reading the pre-trained model.")

    checkpoint = torch.load('vdsr-TB291-fef487db.pth.tar', map_location=config.device)
    print("CHKPT: ", checkpoint.keys())

    # Load checkpoint state dict. Extract the fitted model weights
    model_state_dict = model.state_dict()
    new_state_dict = {k: v for k, v in checkpoint["state_dict"].items() if k in model_state_dict}

    # Overwrite the pretrained model weights to the current model
    model_state_dict.update(new_state_dict)
    model.load_state_dict(model_state_dict)

    # Load the optimizer model
    optimizer.load_state_dict(checkpoint["optimizer"])

    # Load the scheduler model
    scheduler.load_state_dict(checkpoint["scheduler"])
    print("Loaded pretrained model weights.")

    # Create a folder of super-resolution experiment results
    samples_dir = os.path.join("samples", config.exp_name)
    results_dir = os.path.join("results", config.exp_name)
    if not os.path.exists(samples_dir):
        os.makedirs(samples_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Create training process log file
    writer = SummaryWriter(os.path.join("samples", "logs", config.exp_name))

    # Initialize the gradient scaler
    scaler = amp.GradScaler()

    # Training, validation and reporting accuracy to NNI
    # Initialize training to generate network evaluation indicators
    best_psnr = 0.0
    perf_HPO = True

    for epoch in range(0, config.epochs):
        train(model, train_prefetcher, psnr_criterion, pixel_criterion, optimizer, epoch, scaler, writer)
        _ = validate(model, valid_prefetcher, psnr_criterion, epoch, writer, "Valid")
        psnr = validate(model, test_prefetcher, psnr_criterion, epoch, writer, "Test")
        print("\n")

        # Reporting intermediate psnr to nni
        if perf_HPO:
            nni.report_intermediate_result(psnr)

        # Update lr
        scheduler.step()

        # Automatically save the model with the highest index
        is_best = psnr > best_psnr
        best_psnr = max(psnr, best_psnr)

        # torch.save({"epoch": epoch + 1,
        #             "best_psnr": best_psnr,
        #             "state_dict": model.state_dict(),
        #             "optimizer": optimizer.state_dict(),
        #             "scheduler": scheduler.state_dict()},
        #            os.path.join(samples_dir, f"epoch_{epoch + 1}.pth.tar"))

        # if is_best:
        #     shutil.copyfile(os.path.join(samples_dir, f"epoch_{epoch + 1}.pth.tar"), os.path.join(results_dir, "best.pth.tar"))

        # if (epoch + 1) == config.epochs:
        #     shutil.copyfile(os.path.join(samples_dir, f"epoch_{epoch + 1}.pth.tar"), os.path.join(results_dir, "last.pth.tar"))

    # Reporting final results
    if perf_HPO:
        nni.report_final_result(best_psnr)

    print("Finish !")





