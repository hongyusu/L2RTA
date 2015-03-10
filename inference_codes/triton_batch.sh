#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --time=0-00:00:15 --mem-per-cpu=4000
#SBATCH --array=1-5
#SBATCH -o terminal.out
#SBATCH -p play
#SBATCH --exclusive

export OMP_PROC_BIND=true
module load matlab
python triton_auto_run_RSTA.py $SLURM_ARRAY_TASK_ID $TMPDIR

