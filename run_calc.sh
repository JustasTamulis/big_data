#!/bin/sh
#SBATCH --account=juta1001_mif ## -> tavo allocation‘as
#SBATCH --job-name=A_50
#SBATCH --output=results/lab1/slurm-out-%j.out
#SBATCH --error=results/lab1/slurm-out-%j.err
#SBATCH --partition=main ##->cpu/gpu?????
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=6-0
echo abcd
echo $PATH
echo $SLURM_JOB_ID
echo $SLURM_JOB_NAME
source .venv/bin/activate
python lab1/test_hpc.py