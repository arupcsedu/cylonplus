#!/bin/bash

# Activate the Conda environment
source activate cyp-venv

# Change directory
cd /home/jovyan/work/cylonplus/src/model

# Run the Python script
python single-gpu-cnn.py

# Deactivate the Conda environment (optional)
conda deactivate
