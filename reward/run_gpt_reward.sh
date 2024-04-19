#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=thorirhh21@ru.is        # for example uname@hi.is
#SBATCH --job-name=gpt_rm
#SBATCH --nodes=1
#SBATCH --partition=gpu-2xA100
#SBATCH --time=04-04:00:00                   # run for 4 day maximum
#SBATCH --output=gpt_rm_output.log
#SBATCH --error=gpt_rm_errors.log           # Logs if job crashes


# Activate your virtual environment (if using one)
eval "$(conda shell.bash hook)"
conda activate nlp

# Change to the directory containing your Python script
cd /users/home/thorirhh21/nlp/summary/reward_model/

# Run your Python script
python gpt_reward_model.py
