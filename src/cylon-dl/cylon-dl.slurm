#!/bin/bash
#SBATCH --nodes=2
##SBATCH --ntasks-per-node=37
#SBATCH --time=0:50:00
##SBATCH --partition=parallel
#SBATCH --partition=bii-gpu
#SBATCH --gres=gpu:v100:2
#SBATCH -A bii_dsc_community
#SBATCH --output=/project/bii_dsc_community/djy8hg/arupcsedu/cylonplus/src/cylon-dl/logs/cylon-dl-%x-%j.out
#SBATCH --error=/project/bii_dsc_community/djy8hg/arupcsedu/cylonplus/src/cylon-dl/logs/cylon-dl-%x-%j.err


module load anaconda


source activate /scratch/djy8hg/env/gcylon_env
export PATH=$CONDA_PREFIX/bin:$PATH LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH PYTHONPATH=$CONDA_PREFIX/lib/python3.11/site-packages

echo $CONDA_PREFIX

#mpirun -np 4 python cylon_pytorch_dl.py
python cylon_pytorch_dl.py
