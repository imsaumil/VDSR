# Importing required libraries
from train import *
from knowledge_distill import KnowledgeDistill


pruned_model_path = f"generated_models/vdsr_pruned_{config.upscale_factor}.torch"
unpruned_model_path = f"generated_models/vdsr_unpruned_{config.upscale_factor}.torch"

print("\nPerforming distillation...\n")

teacher_model = torch.load(unpruned_model_path, map_location=config.device)
student_model = torch.load(pruned_model_path, map_location=config.device)

kd = KnowledgeDistill(teacher_model=teacher_model, kd_T=5)

alpha = 1
beta = 0.8

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