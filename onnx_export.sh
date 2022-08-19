#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=2-00:00:00
#SBATCH --job-name=truppr20220811
#SBATCH --partition=ce-mri
#SBATCH --gres=gpu:v100:2
#SBATCH --cpus-per-task=48
#SBATCH --output=%j.log

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.conda/envs/gnt/lib/
# python3 train.py --config configs/gnt_blender.txt --train_scenes chair --eval_scenes chair
python3 train.py --onnx --coreml --ckpt_path "./out/gnt_blender/model_250000.pth" --config configs/gnt_blender.txt --train_scenes chair --eval_scenes chair

