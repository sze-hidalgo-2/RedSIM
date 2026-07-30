#!/bin/bash -l
#SBATCH --job-name=redsim
#SBATCH --partition=standard
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --account=project_465002685

module load LUMI
module load partition/C
module load cpeGNU

srun --cpu-bind=verbose,map_cpu:0,16,32,48,64,80,96,112 "$1"
