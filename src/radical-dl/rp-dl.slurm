#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=30
#SBATCH --time=0:30:00
#SBATCH --partition=gpu
#SBATCH -A bii_dsc_community
#SBATCH --output=rp-%x-%j.out
#SBATCH --error=rp-%x-%j.err
#SBATCH --gres=gpu:a100:1


module load anaconda
module load intel

source /scratch/upy9gr/workdir/rp_dl/bin/activate

export RADICAL_LOG_LVL="DEBUG"
export RADICAL_PROFILE="TRUE"

python /scratch/upy9gr/workdir/cylonplus/src/model/single-gpu-cnn-radical-pilot.py uva.rivanna