#!/bin/bash
#SBATCH --job-name=run       # Job name
#SBATCH --output=run.log     # Output file
#SBATCH --error=run.log      # Error file
#SBATCH --partition=gpu      # Partition (queue) name
#SBATCH --nodes=1            # Number of nodes
#SBATCH --ntasks=1           # Number of tasks (divide your total CPUs)
#SBATCH --cpus-per-task=1    # Number of CPU cores per task (total CPUs divided by tasks)
#SBATCH --mem=4G             # Memory per node
#SBATCH --time=00:30:00      # Walltime

PWD=`pwd`
TARGET=$PWD/target/uvacs

echo "PWD: $PWD"
echo "TARGET: $TARGET"

source $TARGET/activate.sh 
source $TARGET/run.sh