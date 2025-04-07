#!/bin/sh
#SBATCH --account=juta1001_mif ## -> tavo allocation‘as
#SBATCH --job-name=Justas_big_data
#SBATCH --output=results/lab1/slurm-out-%j.out
#SBATCH --error=results/lab1/slurm-out-%j.err
#SBATCH --partition=main ##->cpu/gpu?????
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
echo "start"
nproc --all
free -h
echo "PATH: $PATH"
echo "PYTHONPATH: $PYTHONPATH"
echo "SLURM_JOB_NAME: $SLURM_JOB_NAME"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "SLURM_NTASKS: $SLURM_NTASKS"
echo "SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK"
echo "SLURM_JOB_CPUS_PER_NODE: $SLURM_JOB_CPUS_PER_NODE"
echo "Current working directory: $(pwd)"
## pip3 install -r requirements.txt
## pip3 install pandas
cd lab1
## python3 detection.py
python3 performance_analysis.py
echo "Done"