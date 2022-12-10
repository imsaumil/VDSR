# Importing required libraries
from torch.optim import Adam
from nni.compression.pytorch.pruning import L1NormPruner, LevelPruner
from nni.algorithms.compression.pytorch.pruning import L1FilterPruner
from nni.compression.pytorch.speedup import ModelSpeedup
import torch
import time

from model import VDSR
from train import *

import utils

# class DistillKL(nn.Module):
#     """Distilling the Knowledge in a Neural Network"""

#     def __init__(self, T):
#         super(DistillKL, self).__init__()
#         self.T = T

#     def forward(self, y_s, y_t):
#         loss = psnr_criterion(y_s,y_t)
#         # loss = pixel_criterion(y_s,y_t)
#         # p_s = torch.nn.functional.log_softmax(y_s / self.T, dim=1)
#         # p_t = torch.nn.functional.softmax(y_t / self.T, dim=1)
#         # loss = torch.nn.functional.kl_div(p_s, p_t, size_average=False) * (self.T**2) / y_s.shape[0]
#         return loss

def fine_tune(models, optimizer,kd_temperature,train_prefetcher):
    model_s = models[0].train()
    model_t = models[-1].eval()
    # scaler = amp.GradScaler()
    # cri_cls = criterion
    # cri_kd = DistillKL(kd_temperature)

    batches = len(train_prefetcher)
    # Put the generator in training mode

    batch_index = 0

    # Calculate the time it takes to test a batch of data
    end = time.time()

    # Enable preload
    train_prefetcher.reset()
    batch_data = train_prefetcher.next()

    while batch_data is not None:
        # Measure data loading time
        lr = batch_data["lr"].to(config.device, non_blocking=True)
        hr = batch_data["hr"].to(config.device, non_blocking=True)

        # Initialize the generator gradient
        optimizer.zero_grad()

        # Mixed precision training
        with amp.autocast():
            sr_student = model_s(lr)
            sr_teacher = model_t(lr)
            # loss = cri_kd(sr_student, sr_teacher)
            loss = psnr_criterion(sr_student, sr_teacher)

        # Gradient zoom + gradient clipping
        scaler.scale(loss).backward()
        # scaler.update()
        # loss.backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model_s.parameters(), max_norm=config.clip_gradient / optimizer.param_groups[0]["lr"], norm_type=2.0)

        # Update generator weight
        scaler.step(optimizer)
        # optimizer.step()
        scaler.update()

        # Preload the next batch of data
        batch_data = train_prefetcher.next()
        batch_index += 1

if __name__ == '__main__':

    # Best parameters chosen from HPO
    params = {
        "batch_size": 512,
        "lr": 0.0047517718546921175,
        "epochs": 10,
        "momentum": 0.8431304996613767
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

    # define a dummy input size
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
    # best_psnr = 0.0
    psnr = validate(model, test_prefetcher, psnr_criterion, 1, writer, "Test")
    print("\nTHE PSNR OF VANILLA MODEL: ", psnr, "\n\n")
    vanilla_model_path = f"generated_models/vdsr_vanilla_{config.upscale_factor}.torch"
    torch.save(model, vanilla_model_path)
    
    utils.torch2onnx(vanilla_model_path,dummy_input)

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
    
    utils.torch2onnx(unpruned_model_path,dummy_input)

    

    # Defining the configuration list for pruning
    configuration_list = [{
        'sparsity_per_layer': 0.4,
        'op_types': ['Conv2d']
    }, {
        'exclude': True,
        'op_names': ['conv1', 'conv2']
    }]

    # Wrapping the network with pruner
    pruner = L1NormPruner(model, configuration_list)
    print("PRUNER WRAPPED MODEL WITH {}: \n\n".format("L1NormPruner"), model, "\n\n")

    # Next, compressing the model and generating masks
    _, masks = pruner.compress()
    # show the masks sparsity
    for name, mask in masks.items():
        print(name, ' sparsity : ', '{:.2}'.format(mask['weight'].sum() / mask['weight'].numel()), "\n")

    # Need to unwrap the model before speeding-up.
    pruner._unwrap_model()
    
    ModelSpeedup(model, dummy_input.to(config.device), masks).speedup_model()

    print("\nPRUNED MODEL WITH {}: \n\n".format("L1NormPruner"), model, "\n\n")

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
    utils.torch2onnx(pruned_model_path,dummy_input)


    print("\nPerforming distillation\n")
    
    teacher_model = torch.load(unpruned_model_path, map_location=config.device)
    student_model = torch.load(pruned_model_path, map_location=config.device)
    
    models = [student_model, teacher_model]
    optimizer = define_optimizer(student_model, params['lr'], params['momentum'])

    # optimizer = torch.optim.SGD(student_model.parameters(), lr=1e-3)
    start_time = time.time()
    for epoch in range(params['epochs']):
        fine_tune(models,optimizer,5, train_prefetcher) # Set temperature to 5 for distillKL
        psnr = validate(models[0], test_prefetcher, psnr_criterion, epoch, writer, "Test")
    end_time = time.time()

    print("\nTHE TOTAL EXECUTION TIME OF MODEL DISTILLATION: ", end_time-start_time, "\n\n")
    print("\nTHE PSNR OF DISTILLED MODEL: ", psnr, "\n\n")

    distilled_model_path = f"generated_models/vdsr_distilled_{config.upscale_factor}.torch"
    torch.save(model, distilled_model_path)
    utils.torch2onnx(distilled_model_path,dummy_input)