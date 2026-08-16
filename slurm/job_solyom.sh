#!/bin/bash -l
#SBATCH --job-name=redsim
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=16

srun --cpu-bind=verbose,rank_ldom ./redsim_cpu "$1"

