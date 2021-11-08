#!/bin/bash
#SBATCH -J CTRE
#SBATCH --output=slurm-%j_%x.out



module load nvidia/cuda-9.0
module load  nvidia/cuda-9.0_cudnn-7.4 
echo "Gpu devices: "$CUDA_VISIBLE_DEVICES

set -e
set -x

## algorithms  ctre_sim, ctre_seq
## datasets
python3 ./code/main.py 	--dataset_name "madelon"  --train_alg ctre_sim --batch_size 100  \
							--lr 0.01 --epochs 500 --zeta 0.2 --epsilon 1 \
							--num_hidden 1000 --seed 0 --weight_init "random_uniform" \
							--lr_drop_epoch 500


