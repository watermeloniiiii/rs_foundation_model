#!/usr/bin/env bash

#SBATCH --job-name=dinov2
#SBATCH --partition=public
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=10
#SBATCH --mem=120G
#SBATCH --export=ALL



VENV_DIR=/NAS3/Members/linchenxi/rs_foundation_model/venv
CHDIR=/NAS3/Members/linchenxi/rs_foundation_model

export PYTHONPATH=$VENV_DIR/lib/python3.9/site-packages:$PYTHONPATH:/NAS6/Members/linchenxi/satellite-platform-gitlab:/NAS3/Members/linchenxi/dinov2
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
export PATH=$PATH:$CHDIR
cd $CHDIR

source $VENV_DIR/bin/activate
echo "$(which python3)"

export MASTER_PORT=12340
echo "NODELIST="${SLURM_NODELIST}
echo "jobnode_lst="${SLURM_JOB_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

export MASTER_ADDR=$master_addr
echo "master addr="${MASTER_ADDR}
srun bash main/accelerate_launch.sh

# bash run_your_job.sh
# change "public" to v100 to use v100