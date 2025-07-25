#!/bin/bash -e
#SBATCH --job-name=01_prep_occ # job name (shows up in the queue)
#SBATCH --time=01:00:00      # Walltime (HH:MM:SS)
#SBATCH --mem-per-cpu=8GB
#SBATCH --ntasks 1
#SBATCH --nodes 1

#SBATCH --account=uc03610

#SBATCH -o logs/01_prep_%j.out
#SBATCH -e logs/01_prep_%j.err

# Activate the conda environment

mkdir -p logs

echo 'Purging modules and loading Miniconda'
module purge && module load Miniconda3

echo 'Sourcing conda'
source $(conda info --base)/etc/profile.d/conda.sh
export PYTHONNOUSERSITE=1

echo 'Loading postal environment'
conda activate postal

echo 'Running with prep stage SLURM parameters'

if [ `grep "calculate_fault_model_PPE = " run_aggregate_weighted_branches.py | head -n 1 | grep "calculate_fault_model_PPE = True" | wc -l` == 1 ]; then
	if [ `grep "nesi_step = 'prep'" run_aggregate_weighted_branches.py | wc -l` == 1 ]; then
		echo 'python run_aggregate_weighted_branches.py'
		python run_aggregate_weighted_branches.py
	else
		echo "Fails: Set nesi_step to 'prep'"
	fi
else
	echo "Fails: Set calculate_fault_model_PPE to True"
fi


# to call:
# sbatch slurm_example.sl
