#!/bin/bash -e
#SBATCH --job-name=06_wprep # job name (shows up in the queue)
#SBATCH --time=01:00:00      # Walltime (HH:MM:SS)
#SBATCH --mem-per-cpu=2GB
#SBATCH --ntasks 1
#SBATCH --nodes 1

#SBATCH --account=uc03610

#SBATCH -o logs/06_wprep_%j.out
#SBATCH -e logs/06_wprep_%j.err

# Activate the conda environment

mkdir -p logs

echo 'Purging modules and loading Miniconda'
module purge && module load Miniconda3

echo 'Sourcing conda'
source $(conda info --base)/etc/profile.d/conda.sh
export PYTHONNOUSERSITE=1

echo 'Loading postal environment'
conda activate postal

echo 'Running with prep weighted stage SLURM parameters'

if [ `grep "nesi_step = 'prep'" run_aggregate_weighted_branches.py | wc -l` == 1 ]; then
	if [ `grep "calculate_fault_model_PPE = False" run_aggregate_weighted_branches.py | wc -l` == 1 ]; then
		if [ `grep "calculate_weighted_mean_PPE = True" run_aggregate_weighted_branches.py | wc -l` == 1 ]; then
			echo 'python run_aggregate_weighted_branches.py'
			python run_aggregate_weighted_branches.py
		else
			echo "Fails: Set calculate_weighted_mean_PPE to True"
		fi
	else
		 echo "Fails: Set calculate_fault_model_PPE to False"
	fi
else
	echo "Fails: Set nesi_step to 'prep'"
fi


# to call:
# sbatch slurm_example.sl
