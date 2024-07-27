# Directions to Train CNN Model Using Radical Pilot

Follow these steps to successfully train a CNN model using Radical Pilot on the cluster.

## Step-by-Step Instructions

### 1. Load the Necessary Modules

Ensure you load the necessary modules in your shell environment. This ensures all dependencies and necessary paths are set up correctly.

```bash
module load anaconda intel
```

### 2. Activate Your Virtual Environment

Activate your virtual environment where your Python dependencies are installed.

```bash
source /scratch/upy9gr/workdir/rp_dl/bin/activate
```

### 3. Update Radical.Pilot Library

If the Radical.Pilot library has not been updated on PyPi.org (as of 07/27/24 it has not), you need to clone the Radical.Pilot repository and install the package directly to get the latest updates.

```bash
git clone https://github.com/radical-cybertools/radical.pilot
cd radical.pilot
pip install .
```

### 4. Install MPI for Python (mpi4py)

Ensure that `mpi4py` is installed in your virtual environment to support MPI communication.

```bash
pip install mpi4py
```

### 5. Edit the SLURM Script

Edit the `single-gpu-cnn.slurm` script to include the correct path to your virtual environment and the path to your training script.

Example:
```bash
ENV_PATH=/scratch/upy9gr/workdir/rp_dl
SCRIPT_PATH=/scratch/upy9gr/workdir/cylonplus/src/model/single-gpu-cnn-radical-pilot.py
```

The edited SLURM script should look like this:

```bash
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=30
#SBATCH --time=0:30:00
#SBATCH --partition=gpu
#SBATCH -A bii_dsc_community
#SBATCH --output=rp-%x-%j.out
#SBATCH --error=rp-%x-%j.err
#SBATCH --gres=gpu:a100:1

ENV_PATH=/scratch/upy9gr/workdir/rp_dl
SCRIPT_PATH=/scratch/upy9gr/workdir/cylonplus/src/model/single-gpu-cnn-radical-pilot.py

module load anaconda
module load intel

source $ENV_PATH/bin/activate

export RADICAL_LOG_LVL="DEBUG"
export RADICAL_PROFILE="TRUE"

python $SCRIPT_PATH uva.rivanna
```

### 6. Submit the Job to the Cluster

Submit the SLURM script to the cluster to start training the CNN model.

```bash
sbatch single-gpu-cnn.slurm
```

### 7. Retrieving the Model Output

If you have not changed the output path in the single-gpu-cnn.py script, the model output will be saved in the `radical.pilot.sandbox/SESSION_ID/pilot.0000/task.000000` directory.


## Additional Information

- **Check Job Status:** After submitting the job, you can check the status of your job using `squeue` or `scontrol` commands.
  ```bash
  squeue -u your_username
  ```
- **Debugging:** If the job fails, inspect the `.out` and `.err` files specified in the SLURM script for any error messages or logs that can help you debug the issue.

- **Virtual Environment Management:** Ensure that all dependencies required by your script are installed in your virtual environment to avoid runtime errors.