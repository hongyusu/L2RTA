#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --time=3-00:00:00 --mem-per-cpu=4000
#SBATCH --array=1-1000000
#SBATCH -p play
export OMP_PROC_BIND=true
module load matlab
python triton_auto_run_RSTA.py $SLURM_ARRAY_TASK_ID


