#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=30
#SBATCH --time=0:30:00
#SBATCH --partition=bii-gpu
#SBATCH -A bii_dsc_community
#SBATCH --output=rp-%x-%j.out
#SBATCH --error=rp-%x-%j.err
#SBATCH --gres=gpu:1

ENV_PATH=/scratch/upy9gr/workdir/rp_dl
SCRIPT_PATH=/scratch/upy9gr/workdir/cylonplus/src/radical-dl/neural-forecast/raptor-neural-forecast.py

module load anaconda
module load intel

source $ENV_PATH/bin/activate

export RADICAL_LOG_LVL="DEBUG"
export RADICAL_PROFILE="TRUE"

python $SCRIPT_PATH 









