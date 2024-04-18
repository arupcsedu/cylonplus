# cylonplus
High-Performance Distributed Data frames for Machine Learning/Deep Learning Model

## Installation instructions UVA CS cluster

### Login to cluster

```bash
ssh your_computing_id@gpusrv08 -J your_computing_id@portal.cs.virginia.edu
```

### Setup Cylon

```
ssh your_computing_id@gpusrv08 -J your_computing_id@portal.cs.virginia.edu
git clone https://github.com/arupcsedu/cylonplus.git
cd cylonplus
module load anaconda3

conda create -n cyp-venv python=3.11
conda activate cyp-venv

conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
DIR=/u/$USER/anaconda3/envs/cyp-venv 

export CUDA_HOME=$DIR/bin
export PATH=$DIR/bin:$PATH LD_LIBRARY_PATH=$DIR/lib:$LD_LIBRARY_PATH PYTHONPATH=$DIR/lib/python3.11/site-packages 

pip install petastorm

cd src/model
python multi-gpu-cnn.py

```


## Installation instructions UVA Rivanna cluster

We assume that you are able to ssh into rivanna instead of using the ondemand system. This is easily done by following instructions given on <https://infomall.org>. Make sure to modify your .ssh/config file and add the host rivanna.
If you use Windows we recommand not to use putty but use gitbash as it mimics a bash environment that is typical also for Linux systems and thus we only have to maintaine one documentation.

### Login to cluster

```bash
ssh rivanna
```

### Login into a GPU worker node

```bash
source target/rivanna/activate.sh a100
```

Make sure your ~/.condarc file looks like

 cat ~/.condarc

```
env_prompt: '({name}) '
pkgs_dirs:
  - /scratch/thf2bn/.conda/pkgs
```

change the value of thf2bn to the value of $USER

### Setup a PROJECT dir

We assume you will deplyt the code in /scratch/$USER. Note this directory is not backed up. Make sure to backup your changes regularly elsewhere with rsync or use github.

NOTE: the following is yet untested

```bash
export SCRATCH=/scratch/$USER/workdir
export PROJECT=/scratch/$USER/workdir/cylonplus
mkdir -p $SCRATCH
cd $SCRATCH
```

### Setup Cylonplus

We created two simple scripts. The first removes the coonda environment if existing, the second installs it.

```bash
source target/rivanna/clean.sh
source target/rivanna/install.sh
```

The scripts are available in github at 

* <https://github.com/laszewsk/cylonplus/blob/main/target/rivanna/clean.sh>
* <https://github.com/laszewsk/cylonplus/blob/main/target/rivanna/install.sh>

Once it is installed you can in a shell just activate it so you do not need to reiinstall it all the time with

```bash
source target/rivanna/activate.sh
```

* <https://github.com/laszewsk/cylonplus/blob/main/activate.sh>

### Running the program on the interactivenode

```bash
source target/rivanna/run.sh
```


### Using a slurm script to do the install, activation, and run


```bash
sbatch target/rivanna/run-simple.slurm
```

```bash
squeue --me
```

run-simple.slurm:

```bash
TODO: add the slurm parameters in the script. see rivanna documentation
cd src/model
python multi-gpu-cnn.py
```

submit the script

```bash
sbatch script.slurm
```

or use 

```bash
watch sbatch script.slurm
```

for a continious uppdate every second.