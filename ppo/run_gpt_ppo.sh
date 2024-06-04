#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=thorirhh21@ru.is        # for example uname@hi.is
#SBATCH --job-name=rlhf_gpt
#SBATCH --nodes=1
#SBATCH --partition=gpu-8xA100
#SBATCH --time=04-00:00:00                   # run for 4 day maximum
#SBATCH --output=gpt_sft_ppo_norm_e20_v2_output.log
#SBATCH --error=gpt_sft_ppo_norm_e20_v2_errors.log        # Logs if job crashes


# Activate your virtual environment (if using one)
eval "$(conda shell.bash hook)"
conda activate nlp

# Change to the directory containing your Python script
cd /users/home/thorirhh21/nlp/summary/ppo/

# Run your Python script
python gpt_ppo_norm.py
