#!/bin/sh

# Detect GPU and CPU information
GPU_IDS=$(lspci | grep ' VGA ' | cut -d" " -f 1)
GPU_NUM=$(echo "$GPU_IDS" | wc -w)
CPU_NUM=$(cat /proc/cpuinfo | grep processor | wc -l)

# Set GPU environment variables
export CUDA_VISIBLE_DEVICES="0"
export GPU_DEVICE_ORDINAL="0"

# Get CPU and GPU IDs
CPU_ID=$(cat /proc/$$/stat | cut -f 40 -d ' ')
GPU_ID="$CUDA_VISIBLE_DEVICES"
test -z "$GPU_ID" && GPU_ID="$GPU_DEVICE_ORDINAL"

# Set OpenMP and MPI variables
OMP_NUM="$OMP_NUM_THREADS"
MPI_RANK="$PMI_RANK$PMIX_RANK$ALPS_APP_PE"
NODE=$(hostname)

OMP_NUM=1
test -z "$ALPS_APP_DEPTH"  || OMP_NUM="$ALPS_APP_DEPTH"
test -z "$OMP_NUM_THREADS" || OMP_NUM="$OMP_NUM_THREADS"

MPI_RANK=0
test -z "$ALPS_APP_PE"  || MPI_RANK="$ALPS_APP_PE"
test -z "$PMIX_RANK"    || MPI_RANK="$PMIX_RANK"
test -z "$PMI_RANK"     || MPI_RANK="$PMI_RANK"

# Function to print sequence
seq()  (i=0; while test $i -lt $1; do echo $i; i=$((i+1)); done)

# Check if required variables are set
test -z "$OMP_NUM"  && echo 'OMP_NUM_THREADS / ALPS_APP_DEPTH not set' && exit 1
test -z "$MPI_RANK" && echo 'PMI_RANK / PMIX_RANK / ALPS_APP_PE not set' && exit 1

# Print environment information
for idx in $(seq $OMP_NUM); do (echo "$NODE $MPI_RANK:$idx/$OMP_NUM @ $CPU_ID/$CPU_NUM : $GPU_ID/$GPU_NUM")& done
for idx in $(seq $OMP_NUM); do wait; done

sleep 1
if test "$MPI_RANK" = 0
then
    echo
    echo "OMP_NUM_THREADS: $OMP_NUM_THREADS"
    echo "ALPS_APP_DEPTH : $ALPS_APP_DEPTH"
    echo "PMI_RANK       : $PMI_RANK"
    echo "PMIX_RANK      : $PMIX_RANK"
    echo "ALPS_APP_PE    : $ALPS_APP_PE"
    echo "NODE           : $(hostname)"
fi

echo "Starting PyTorch CNN training..."

# Set up the environment for PyTorch (modify this based on your setup)
# If you're using a module system, you might use something like:
# module load cuda/11.3 cudnn/8.2.0 python/3.8 pytorch/1.9.0
# If you're using a virtual environment, you might use:
# source /path/to/your/pytorch/environment/bin/activate

# Set the number of OpenMP threads
export OMP_NUM_THREADS=$OMP_NUM

# Run the PyTorch script
python single-gpu-cnn.py --epochs 10 --batch-size 64 --test-batch-size 1000 --lr 0.01 --gamma 0.7 --save-model

# If you activated a virtual environment, deactivate it
# deactivate

echo "PyTorch CNN training completed."