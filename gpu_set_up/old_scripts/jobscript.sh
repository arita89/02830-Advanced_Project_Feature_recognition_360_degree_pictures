#!/bin/sh
#python 360_object_detection_draft.py
### General options
#BSUB -q gpuvolta
### -- set the job Name --
#BSUB -J jobscript_arita
### -- ask for number of cores (default: 1) --
#BSUB -n 2
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 1:00
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


# Load the cuda module
module load cuda/10.0

/appl/cuda/10.0/samples/NVIDIA_CUDA-10.2_Samples/bin/x86_64/linux/release/deviceQuery

nvidia-smi
