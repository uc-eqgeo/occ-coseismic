#!/bin/bash -e
#SBATCH --job-name=05_all_dict # job name (shows up in the queue)
#SBATCH --time=01:00:00      # Walltime (HH:MM:SS)
#SBATCH --mem-per-cpu=2GB
#SBATCH --ntasks 1
#SBATCH --nodes 1

#SBATCH --account=uc03610

#SBATCH -o logs/05_all_dict_%j.out
#SBATCH -e logs/05_all_dict_%j.err

# Activate the conda environment

mkdir -p logs

echo 'Purging modules and loading Miniconda'
module purge && module load Miniconda3

echo 'Sourcing conda'
source $(conda info --base)/etc/profile.d/conda.sh
export PYTHONNOUSERSITE=1

conda activate postal

echo 'Running with build PPE stage SLURM parameters'

if [ `grep "calculate_fault_model_PPE = " run_aggregate_weighted_branches.py | head -n 1 | grep "calculate_fault_model_PPE = True" | wc -l` == 1 ]; then
	if [ `grep "prep_sbatch = False" run_aggregate_weighted_branches.py | wc -l` == 1 ]; then
		if [ `grep "calculate_weighted_mean_PPE =" run_aggregate_weighted_branches.py | head -n 1 | grep "calculate_weighted_mean_PPE = False" | wc -l` == 1 ]; then
			if [ `grep "nesi_step = 'combine'" run_aggregate_weighted_branches.py | wc -l` == 1 ]; then
				echo 'python run_aggregate_weighted_branches.py'
				python run_aggregate_weighted_branches.py
			else
				echo "Fails: Set nesi_step to 'combine'"
			fi
		else
			echo "Fails: Set calculate_weighted_mean_PPE to False"
		fi
	else
		echo "Fails: Set prep_sbatch to 'False'"
	fi
else
	echo "Fails: Set calculate_fault_model_PPE to True"
fi

	

# to call:
# sbatch slurm_example.sl
