#!/bin/bash

## This script is designed to use on a system that uses SLURM to schedule jobs.
## If your system doesn't use SLURM, replace line 19 and 23 with appropriate 
## commands.
## INPUTS TO THIS SCRIPT
## (1) String representing dataset name
## (2) String representing kernel name

# Number of allowed jobs in queue for each dataset
num=10

vars=($(seq 0.1 0.1 5))
depths=($(seq 1 1 32))

for var_idx in "${!vars[@]}"
do
    for l_idx in "${!depths[@]}"
    do
        while [ $(squeue -n "${2} ${1}" | wc -l) -gt $num ]
        do
            sleep 1
        done
        sbatch -J "${2} ${1}" grid_iteration.sh $1 $2 ${vars[$var_idx]} $var_idx ${depths[$l_idx]} $l_idx
    done
done
