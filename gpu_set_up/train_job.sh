#!/bin/sh
#python fire_train.py
### General options
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J train_job
### -- ask for number of cores (default: 1) --
#BSUB -n 4
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process:gmodel=Tesla-32GB"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 2:00
# request 5GB of system-memory
#BSUB -R "rusage[mem=5GB]"
### -- set the email address --
#BSUB -u arita@fysik.dtu.dk
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o gpu-%J.out
#BSUB -e gpu_%J.err
# -- end of LSF options --

module load cuda/10.1
module load cudnn/v7.6.5.32-prod-cuda-10.1
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/appl/cuda/10.1/extras/CUPTI/lib64/
export PATH="/zhome/94/5/101974/.local/bin:$PATH" 
source "/zhome/94/5/101974/Desktop/Interactive_Nodes/tf_2-3/bin/activate"
python fire_train.py
