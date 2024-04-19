#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=thorirhh21@ru.is        # for example uname@hi.is
#SBATCH --job-name=create_dataset
#SBATCH --nodes=1
#SBATCH --partition=gpu-1xA100
#SBATCH --time=04-04:00:00                   # run for 1 day maximum
#SBATCH --output=create_dataset_output.log
#SBATCH --error=create_dataset_errors.log        # Logs if job crashes


# Activate your virtual environment (if using one)
eval "$(conda shell.bash hook)"
conda activate nlp

# Change to the directory containing your Python script
cd /users/home/thorirhh21/nlp/summary/reward_model/

# Run your Python script
python create_reward_data.py
