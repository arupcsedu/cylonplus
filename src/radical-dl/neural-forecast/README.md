# Using NeuralForecast with Radical Pilot on HPC Clusters

This guide outlines how to set up and run NeuralForecast models using Radical Pilot on a high-performance computing (HPC) cluster. NeuralForecast is a powerful library for time series forecasting with neural networks, and Radical Pilot allows for efficient distributed computing.

## Prerequisites

- Access to an HPC cluster with SLURM job scheduler or RAPTOR
- Basic knowledge of Python and time series forecasting concepts

## Step-by-Step Setup

### 1. **(Optional) Request Interactive Job on RAPTOR**

If you are using RAPTOR, start by requesting an interactive job using `ijob`:

```bash
ijob -c 1 -A bii_dsc_community -p standard --time=00:30:00
```

If you are **not using RAPTOR**, skip this step and continue with the next step.

### 2. **Load Required Modules**

Load the necessary modules on your cluster:

```bash
module load anaconda
module load intel
```

### 3. **Set Up Virtual Environment**

Create and activate a virtual environment:

```bash
python -m venv /scratch/upy9gr/workdir/rp_dl
source /scratch/upy9gr/workdir/rp_dl/bin/activate
```

### 4. **Install Required Packages**

Install Radical Pilot, NeuralForecast, and other necessary packages:

```bash
pip install radical.pilot neuralforecast matplotlib mpi4py
```

If the latest Radical Pilot is not on PyPI, install it from the repository:

```bash
git clone https://github.com/radical-cybertools/radical.pilot
cd radical.pilot
pip install .
```

### 5. **Prepare Your NeuralForecast Script**

Create a Python script (e.g., `time-series-forecast.py`) that uses NeuralForecast. Here's a basic example:

```python
from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS
from neuralforecast.utils import AirPassengersDF

# Create a NeuralForecast instance with the NBEATS model
nf = NeuralForecast(
    models=[NBEATS(input_size=24, h=12, max_steps=100)],
    freq='M'
)

# Fit the model with the AirPassengers dataset
nf.fit(df=AirPassengersDF)

# Make predictions and print the results
predictions = nf.predict()
print(predictions)
```

### 6. **Run the Script**

Depending on your setup, the next steps will vary:

- **If using RAPTOR**: Directly run the script without a SLURM script. Ensure you’re in the interactive `ijob` environment before running:

    ```bash
    python /scratch/upy9gr/workdir/cylonplus/src/radical-dl/neural-forecast/raptor-neural-forecast.py uva.rivanna
    ```

- **If using a SLURM-based cluster**: Create a SLURM script (e.g., `time-series-forecast.slurm`) to submit your job:

    ```bash
    #!/bin/bash
    #SBATCH --nodes=1
    #SBATCH --ntasks-per-node=30
    #SBATCH --time=0:30:00
    #SBATCH --partition=gpu
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

    python $SCRIPT_PATH uva.rivanna
    ```

    Submit your job to the cluster:

    ```bash
    sbatch time-series-forecast.slurm
    ```

### 7. **Monitor and Retrieve Results**

- **If using RAPTOR**: Results and output will be available directly in the terminal once the script completes.
- **If using SLURM**: Check your job status with:

    ```bash
    squeue -u your_username
    ```

    Once complete, check the output files (`rp-%x-%j.out` and `rp-%x-%j.err`) for results and any error messages.

### Advanced Usage

- **Distributed Training**: For large datasets or complex models, you can leverage Radical Pilot's capabilities to distribute your NeuralForecast workload across multiple nodes.
- **Hyperparameter Tuning**: Use Radical Pilot to manage parallel runs for hyperparameter optimization of your NeuralForecast models.
- **Ensemble Forecasting**: Implement ensemble methods by running multiple NeuralForecast models in parallel and combining their predictions.

### Troubleshooting

- If you encounter memory issues, try adjusting the `--mem` parameter in your SLURM script or reducing the `input_size` parameter in NeuralForecast models.
- For GPU acceleration, add `#SBATCH --gres=gpu:1` to your SLURM script (for SLURM setups) or allocate GPU resources during iJob initialization on RAPTOR.

### Additional Resources

- [NeuralForecast Documentation](https://nixtla.github.io/neuralforecast/)
- [Radical Pilot Documentation](https://radicalpilot.readthedocs.io/)

Remember to adjust paths, account information, and resource requests in the SLURM script or iJob command according to your specific cluster configuration and needs.