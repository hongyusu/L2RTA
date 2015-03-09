#!/bin/bash
#SBATCH --time=4:00:00 --mem-per-cpu=2000
#SBATCH --array=1-616
module load matlab
echo $SLURM_ARRAY_TASK_ID
#matlab -nojvm -r "run_met_id($SLURM_ARRAY_TASK_ID)"

