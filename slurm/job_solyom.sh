#!/bin/bash -l
#SBATCH --job-name=redsim
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=16

module load openmpi/4.1.2
srun --cpu-bind=verbose,rank_ldom ./redsim_cpu "$1"

