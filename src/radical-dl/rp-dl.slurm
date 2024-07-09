#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --time=0:50:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:2
#SBATCH -A bii_dsc_community
#SBATCH --output=/scratch/upy9gr/workdir/cylonplus/src/radical-dl/logs/rp-dl-%x-%j.out
#SBATCH --error=/scratch/upy9gr/workdir/cylonplus/src/radical-dl/logs/rp-dl-%x-%j.err


module load anaconda


source /scratch/upy9gr/workdir/rp_dl/bin/activate

export RADICAL_LOG_LVL="DEBUG"
export RADICAL_PROFILE="TRUE"

python hello_world_simple.py