# Importing required libraries
from nni.compression.pytorch.pruning import FPGMPruner, LevelPruner
from nni.compression.pytorch.speedup import ModelSpeedup
from torch.utils.tensorboard import SummaryWriter

from model import VDSR
from train import *

import os
import utils

epochs = 10


if __name__ == '__main__':

    # Best parameters chosen from HPO
    params = {
        "batch_size": 1028,
        "lr": 0.0026923044194619525,
        "epochs": 10,
        "momentum": 0.1629420704114407
    }

    # Using pre-fetcher to load data into memory
    train_prefetcher, valid_prefetcher, test_prefetcher = load_dataset(params['batch_size'])
    print("Load train dataset and valid dataset successfully.")

    # Creating the unpruned model
    model = VDSR().to(config.device)
    print("Build VDSR model successfully.\n")
    print("ORIGINAL UN-PRUNED MODEL: \n", model, "\n\n")

    # Defining the loss function
    psnr_criterion, pixel_criterion = define_loss()
    print("Defined all loss functions successfully.")

    # Defining the optimizer to reduce the losses
    optimizer = define_optimizer(model, params['lr'], params['momentum'])
    print("Defined optimizer functions successfully.")

    # Defining the optimizer scheduler
    scheduler = define_scheduler(optimizer)
    print("Defined all optimizer scheduler successfully.")

    # Define a dummy input size
    dummy_input = torch.rand(64, 1, 28, 28).to(config.device)

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

    # Update the learning rate and momentum to the optimal ones,
    # keeping the other params same
    for g in optimizer.param_groups:
        g['lr'] = params['lr']
        g['momentum'] = params['momentum']

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
    psnr = validate(model, test_prefetcher, psnr_criterion, 1, writer, "Test")
    print("\nTHE PSNR OF VANILLA MODEL: ", psnr, "\n\n")

    vanilla_model_path = f"generated_models/vdsr_vanilla_{config.upscale_factor}.torch"
    torch.save(model, vanilla_model_path)

    utils.torch2onnx(vanilla_model_path, dummy_input)

    # Starting time for the unpruned model
    start_time = time.time()

    for epoch in range(0, params['epochs']):
        train(model, train_prefetcher, psnr_criterion, pixel_criterion, optimizer, epoch, scaler, writer)
        _ = validate(model, valid_prefetcher, psnr_criterion, epoch, writer, "Valid")
        psnr = validate(model, test_prefetcher, psnr_criterion, epoch, writer, "Test")
        print("\n")

        # Update lr
        scheduler.step()

    # Ending time for unpruned model
    end_time = time.time()

    # The total execution time of unpruned model
    exec_time = end_time - start_time
    print("\nTHE TOTAL EXECUTION TIME OF UNPRUNED MODEL: ", exec_time, "\n\n")
    print("\nTHE PSNR OF UNPRUNED MODEL: ", psnr, "\n\n")

    unpruned_model_path = f"generated_models/vdsr_unpruned_{config.upscale_factor}.torch"
    torch.save(model, unpruned_model_path)

    utils.torch2onnx(unpruned_model_path, dummy_input)

    # Defining the configuration list for pruning
    configuration_list = [{
        'sparsity_per_layer': 0.4,
        'op_types': ['Conv2d']
    }, {
        'exclude': True,
        'op_names': ['conv1', 'conv2']
    }]

    # Wrapping the network with pruner
    pruner = LevelPruner(model, configuration_list)
    print("PRUNER WRAPPED MODEL WITH {}: \n\n".format("LevelPruner"), model, "\n\n")

    # Next, compressing the model and generating masks
    _, masks = pruner.compress()
    # show the masks sparsity
    for name, mask in masks.items():
        print(name, ' sparsity : ', '{:.2}'.format(mask['weight'].sum() / mask['weight'].numel()), "\n")

    # Need to unwrap the model before speeding-up.
    pruner._unwrap_model()

    ModelSpeedup(model, dummy_input.to(config.device), masks).speedup_model()

    print("\nPRUNED MODEL WITH {}: \n\n".format("LevelPruner"), model, "\n\n")

    # Starting time for pruned model
    start_time = time.time()

    for epoch in range(0, params['epochs']):
        train(model, train_prefetcher, psnr_criterion, pixel_criterion, optimizer, epoch, scaler, writer)
        _ = validate(model, valid_prefetcher, psnr_criterion, epoch, writer, "Valid")
        psnr = validate(model, test_prefetcher, psnr_criterion, epoch, writer, "Test")
        print("\n")

        # Update lr
        scheduler.step()

    # Ending time for pruned model
    end_time = time.time()

    # The total execution time of pruned model
    exec_time = end_time - start_time
    print("\nTHE TOTAL EXECUTION TIME OF PRUNED MODEL: ", exec_time, "\n\n")
    print("\nTHE PSNR OF PRUNED MODEL: ", psnr, "\n\n")

    pruned_model_path = f"generated_models/vdsr_pruned_{config.upscale_factor}.torch"
    torch.save(model, pruned_model_path)
    utils.torch2onnx(pruned_model_path, dummy_input)