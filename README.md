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
DIR=/u/djy8hg/anaconda3/envs/cyp-venv 

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

### Setup a PROJECT dir

We assume you will deplyt the code in /scratch/$USER. Note this directory is not backed up. Make sure to backup your changes regularly elsewhere with rsync or use github.

NOTE: the following is yet untested

```bash
export SCRATCH=/scratch/$USER/workdir
export PROJECT=/scratch/$USER/workdir/cylonplus
mkdir -p $SCRATCH
cd $SCRATCH
```

### Setup Cylon

TODO: bets to convert to slurm script and run on node as batch script

```bash
rivanna>
  git clone https://github.com/arupcsedu/cylonplus.git
  cd cylonplus
  module load anaconda3

  conda create --prefix=$PROJECT/CYLONPLUS cyp-venv python=3.11
  conda activate --prefix=$PROJECT/CYLONPLUS

  conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

  export PYTHON_DIR=$PROJECT/CYLONPLUS #?
  export CUDA_HOME=$PYTHON_DIR/bin # should that be cuda prefix?
  export PATH=$PYTHON_DIR/bin:$PATH
  export LD_LIBRARY_PATH=$PYTHON_DIR/lib:$LD_LIBRARY_PATH
  export PYTHONPATH=$PYTHON_DIR/lib/python3.11/site-packages 

  pip install petastorm
```

once converted to a slurm script called `install.slurm` execute

```bash
sbatch install.slurm
```

### Running the program

create a slurm script that includes 

script.slurm:

```bash
TODO: add the slurm parameters in the script. see rivanna documentation
cd src/model
python multi-gpu-cnn.py
```

submit the script

```bash
sbatch script.slurm
```
