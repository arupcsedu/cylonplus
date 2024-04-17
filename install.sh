export SCRATCH=/scratch/$USER/workdir
export PROJECT=/scratch/$USER/workdir/cylonplus

cd $PROJECT
module purge
module load anaconda

conda create --prefix=$PROJECT/CYLONPLUS python=3.11 -y
conda activate $PROJECT/CYLONPLUS

conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

export PYTHON_DIR=$PROJECT/CYLONPLUS
export CUDA_HOME=$PYTHON_DIR/bin 
export PATH=$PYTHON_DIR/bin:$PATH
export LD_LIBRARY_PATH=$PYTHON_DIR/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$PYTHON_DIR/lib/python3.11/site-packages 

pip install petastorm
pip install cloudmesh-common