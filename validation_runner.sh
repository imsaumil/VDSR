#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p GPU-shared
#SBATCH -t 10:00:00
#SBATCH --gres=gpu:1

source ~/.bashrc
module load cuda/11.7.1
module load anaconda3

conda activate env2

cd /jet/home/jshah2/VDSR/
# nnictl view
python3 validate.py generated_models/vdsr_vanilla_4.torch
python3 validate.py generated_models/vdsr_unpruned_4.torch
python3 validate.py generated_models/vdsr_pruned_4.torch
python3 validate.py generated_models/vdsr_distilled_4.torch