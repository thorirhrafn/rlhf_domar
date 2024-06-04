#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=thorirhh21@ru.is        # for example uname@hi.is
#SBATCH --job-name=dpo_llama
#SBATCH --nodes=1
#SBATCH --partition=gpu-1xA100
#SBATCH --time=04-04:00:00                   # run for 4 day maximum
#SBATCH --output=llama_dpo_output.log
#SBATCH --error=llama_dpo_errors.log        # Logs if job crashes


# Activate your virtual environment (if using one)
eval "$(conda shell.bash hook)"
conda activate nlp

# Change to the directory containing your Python script
cd /users/home/thorirhh21/nlp/summary/dpo/

# Run your Python script
python llama_dpo.py
