#!/bin/bash
sbatch --nodes=1 -o "nodes_$1_1.out" job_lumi.sh "$2"
sbatch --nodes=2 -o "nodes_$1_2.out" job_lumi.sh "$2"
sbatch --nodes=4 -o "nodes_$1_4.out" job_lumi.sh "$2"
sbatch --nodes=8 -o "nodes_$1_8.out" job_lumi.sh "$2"
sbatch --nodes=16 -o "nodes_$1_16.out" job_lumi.sh "$2"
sbatch --nodes=32 -o "nodes_$1_32.out" job_lumi.sh "$2"
sbatch --nodes=64 -o "nodes_$1_64.out" job_lumi.sh "$2"
