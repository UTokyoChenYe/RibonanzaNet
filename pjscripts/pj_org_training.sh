#!/bin/sh

#------ pjsub option --------# 
#PJM -L rscgrp=share
#PJM -L gpu=1
#PJM -L elapse=48:00:00 
#PJM -g gs58
#PJM -j


#------- Program execution -------#
module load aquarius
module load cuda
module load gcc
module load ompi
export MPLCONFIGDIR="/work/gs58/d58004/tmp/matplotlib"
export WANDB_CONFIG_DIR="/work/gs58/d58004/tmp/wandb"
export PATH="/work/02/gs58/d58004/mambaforge/envs/torch/bin/:$PATH"
nvidia-smi
cd /work/gs58/d58004/ideas/RibonanzaNet
accelerate launch run.py --config_path configs/pairwise.yaml