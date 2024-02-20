#!/usr/bin/env bash

#SBATCH --job-name=crop_inference
#SBATCH --partition=public
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=80
#SBATCH --mem=300G
#SBATCH --export=ALL


VENV_DIR=/NAS6/Members/linchenxi/satellite-platform-gitlab/venv
CHDIR=/NAS6/Members/linchenxi/morocco

export PYTHONPATH=$VENV_DIR/lib/python3.9/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
export PATH=$PATH:$CHDIR
cd $CHDIR

source $VENV_DIR/bin/activate
echo "$(which python3)"

python3 main.py

# bash run_your_job.sh
# change "public" to v100 to use v100