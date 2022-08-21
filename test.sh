#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=2-00:00:00
#SBATCH --job-name=truppr20220811
#SBATCH --partition=ce-mri
#SBATCH --gres=gpu:v100:2
#SBATCH --cpus-per-task=48
#SBATCH --output=%j.log

# python3 train.py --config configs/gnt_blender.txt --train_scenes chair --eval_scenes chair
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.conda/envs/gnt/lib/
python3 eval.py --config configs/gnt_blender.txt --train_scenes chair --eval_scenes chair

