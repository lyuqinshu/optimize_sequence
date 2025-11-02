#!/bin/bash
#SBATCH -p gpu
#SBATCH -c 1
#SBATCH -t 3-0:00:00 
#SBATCH --mem=8G
#SBATCH -o py_%j.o 
#SBATCH -e py_%j.e
#SBATCH -p gpu
#SBATCH -n 1
#SBATCH --gres=gpu:1

module purge
module load python
eval "$(mamba shell hook --shell bash)"
mamba activate RSC_sim

cd $home
cd optimize_sequence/all
python optimize_all.py
