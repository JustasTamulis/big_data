# !/bin/sh
# SBATCH --account=juta1001 ## -> tavo allocation‘as
# SBATCH --job-name=A_50
# SBATCH --output=../Results/slurm-out-%j.out
# SBATCH --error=../Results/slurm-out-%j.err
# SBATCH --partition=cpu ##->cpu/gpu?????
# SBATCH --ntasks=1
# SBATCH --cpus-per-task=1
# SBATCH --time=6-0
python3 ./lab1/test_hpc.py