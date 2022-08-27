#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=00:45:00
#SBATCH --job-name=truppr20220811
#SBATCH --partition=ce-mri
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=48
#SBATCH --output=test_evaluation.log


# python3 train.py --config configs/gnt_blender.txt --train_scenes chair --eval_scenes chair
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.conda/envs/gnt/lib/
python3 eval.py --expname lr_lego --config configs/gnt_blender.txt --eval_scenes lr_lego --chunk_size 500 --run_val --N_samples 192 --onnx --coreml
# python3 eval.py --eval_scenes drums --expname gnt_drums --chunk_size 500 --run_val --N_samples 192

