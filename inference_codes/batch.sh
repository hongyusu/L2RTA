#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --time=0:00:01 --mem-per-cpu=2
#SBATCH --array=1-100
module load matlab
echo $SLURM_ARRAY_TASK_ID > tmp
#python triton_auto_run_RSTA.py $SLURM_ARRAY_TASK_ID
#matlab -nojvm -r "run_met_id($SLURM_ARRAY_TASK_ID)"

