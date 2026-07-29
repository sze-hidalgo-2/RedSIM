#!/bin/bash
#SBATCH --job-name=redsim_mpi
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --exclusive

#SBATCH --account=project_465002685
#SBATCH --partition=standard
#SBATCH --time=00:30:00

export OMP_NUM_THREADS=1
srun --cpu-bind=cores ./redsim_cpu "$1"
