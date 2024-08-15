# Using NeuralForecast with Radical Pilot on HPC Clusters

This guide outlines how to set up and run NeuralForecast models using Radical Pilot on a high-performance computing (HPC) cluster. NeuralForecast is a powerful library for time series forecasting with neural networks, and Radical Pilot allows for efficient distributed computing.

## Prerequisites

- Access to an HPC cluster with SLURM job scheduler
- Basic knowledge of Python and time series forecasting concepts

## Step-by-Step Setup

### 1. Load Required Modules

Load the necessary modules on your cluster:

```bash
module load anaconda intel
```

### 2. Set Up Virtual Environment

Create and activate a virtual environment:

```bash
python -m venv /path/to/your/venv
source /path/to/your/venv/bin/activate
```

### 3. Install Required Packages

Install Radical Pilot, NeuralForecast, and other necessary packages:

```bash
pip install radical.pilot neuralforecast matplotlib mpi4py
```

Note: If the latest Radical Pilot is not on PyPI, install it from the repository:

```bash
git clone https://github.com/radical-cybertools/radical.pilot
cd radical.pilot
pip install .
```

### 4. Prepare Your NeuralForecast Script

Create a Python script (e.g., `neuralforecast_example.py`) that uses NeuralForecast. Here's a basic example:

```python
import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS

# Load your time series data
df = pd.read_csv('your_data.csv')

# Initialize and train the model
nf = NeuralForecast(models=[NHITS(input_size=30, h=5)], freq='D')
nf.fit(df)

# Make predictions
forecast = nf.predict()
print(forecast)
```

### 5. Create a SLURM Script

Create a SLURM script (e.g., `run_neuralforecast.slurm`) to submit your job:

```bash
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=30
#SBATCH --time=2:00:00
#SBATCH --partition=standard
#SBATCH -A your_account
#SBATCH --output=nf-%j.out
#SBATCH --error=nf-%j.err

ENV_PATH=/path/to/your/venv
SCRIPT_PATH=/path/to/your/neuralforecast_example.py

module load anaconda intel

source $ENV_PATH/bin/activate

export RADICAL_LOG_LVL="DEBUG"
export RADICAL_PROFILE="TRUE"

python $SCRIPT_PATH
```

### 6. Submit Your Job

Submit your job to the cluster:

```bash
sbatch run_neuralforecast.slurm
```

### 7. Monitor and Retrieve Results

Check your job status:

```bash
squeue -u your_username
```

Once complete, check the output files (`nf-*.out` and `nf-*.err`) for results and any error messages.

## Advanced Usage

- **Distributed Training**: For large datasets or complex models, you can leverage Radical Pilot's capabilities to distribute your NeuralForecast workload across multiple nodes.

- **Hyperparameter Tuning**: Use Radical Pilot to manage parallel runs for hyperparameter optimization of your NeuralForecast models.

- **Ensemble Forecasting**: Implement ensemble methods by running multiple NeuralForecast models in parallel and combining their predictions.

## Troubleshooting

- If you encounter memory issues, try adjusting the `--mem` parameter in your SLURM script.
- For GPU acceleration, add `#SBATCH --gres=gpu:1` to your SLURM script and ensure NeuralForecast is set up to use GPUs.

## Additional Resources

- [NeuralForecast Documentation](https://nixtla.github.io/neuralforecast/)
- [Radical Pilot Documentation](https://radicalpilot.readthedocs.io/)

Remember to adjust paths, account information, and resource requests in the SLURM script according to your specific cluster configuration and needs.