#!/bin/bash -l
#SBATCH --job-name=hand_namd1gpu
#SBATCH --time=1:0:0
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=1G
#SBATCH --gres=gpu:1
#SBATCH -A doconn15_gpu ### Slurm-account is usually the PI userid
#SBATCH --error=err.txt

###Load modules and check
ml purge
module load namd/2.14-cuda-smp
module load anaconda
ml
export CUDA_VISIBLE_DEVICES=0
export CONV_RSH=ssh

###Activate the virtual environment
conda activate /home/ychang73/.conda/envs/lgt-pose

###Run the python script
python scripts/train_hydra.py --config-path=/vast/doconn15/projects/hand_tracking/lightning-pose-main/data/LP_240120 --config-name=config_hand-1cam.yaml

###Deactivate the virtual environment
conda deactivate
