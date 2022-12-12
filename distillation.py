# Importing required libraries
from train import *
from knowledge_distill import KnowledgeDistill
import os
from torch.utils.tensorboard import SummaryWriter

def fine_tune(models, optimizer, kd_temperature, train_prefetcher,scaler):
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
        torch.nn.utils.clip_grad_norm_(model_s.parameters(),
                                       max_norm=config.clip_gradient / optimizer.param_groups[0]["lr"], norm_type=2.0)

        # Update generator weight
        scaler.step(optimizer)
        # optimizer.step()
        scaler.update()

        # Preload the next batch of data
        batch_data = train_prefetcher.next()
        batch_index += 1

if __name__ == '__main__':
    params = {
        "batch_size": 1028,
        "lr": 0.0026923044194619525,
        "epochs": 10,
        "momentum": 0.1629420704114407
    }
    train_prefetcher, valid_prefetcher, test_prefetcher = load_dataset(params['batch_size'])
    print("Load train dataset and valid dataset successfully.")

    pruned_model_path = f"generated_models/vdsr_pruned_{config.upscale_factor}.torch"
    vanilla_model_path = f"generated_models/vdsr_vanilla_{config.upscale_factor}.torch"

    print("\nPerforming distillation...\n")

    teacher_model = torch.load(vanilla_model_path, map_location=config.device)
    student_model = torch.load(pruned_model_path, map_location=config.device)

    kd = KnowledgeDistill(teacher_model=teacher_model, kd_T=5)

    # alpha = 1
    # beta = 0.8

    
    models = [student_model, teacher_model]
    optimizer = define_optimizer(student_model, params['lr'], params['momentum'])
    psnr_criterion, pixel_criterion = define_loss()
    print("Defined all loss functions successfully.")

    scheduler = define_scheduler(optimizer)
    print("Defined all optimizer scheduler successfully.")

    writer = SummaryWriter(os.path.join("samples", "logs", config.exp_name))
    scaler = amp.GradScaler()

    start_time = time.time()
    for epoch in range(params['epochs']):
        fine_tune(models, optimizer, 5, train_prefetcher,scaler)  # Set temperature to 5 for distillKL
        psnr = validate(models[0], test_prefetcher, psnr_criterion, epoch, writer, "Test")
        scheduler.step()

    # Ending time for dostlled model
    end_time = time.time()

    # The total execution time of unpruned model
    exec_time = end_time - start_time
    print("\nTHE TOTAL EXECUTION TIME OF MODEL DISTILLATION: ", exec_time, "\n\n")
    print("\nTHE PSNR OF DISTILLED MODEL: ", psnr, "\n\n")