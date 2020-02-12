#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
###SBATCH --partition=batch

source env/bin/activate

python3 -m code.experiments.03_deep_experiments.grid_iteration $1 $2 $3 $4 $5 $6

deactivate
