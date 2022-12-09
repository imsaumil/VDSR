#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p GPU-shared
#SBATCH -t 10:00:00
#SBATCH --gres=gpu:1

source ~/.bashrc
module load cuda/10.2.0
module load anaconda3

conda activate gpu

cd /jet/home/jshah2/VDSR/
# nnictl view
python3 hpo.py