#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=thorirhh21@ru.is        # for example uname@hi.is
#SBATCH --job-name=gptswe7B
#SBATCH --nodes=1
#SBATCH --partition=gpu-1xA100
#SBATCH --time=06-04:00:00                   # run for 6 day maximum
#SBATCH --output=sft_gpt7b_ft_domar_output.log
#SBATCH --error=sft_gpt7b_ft_domar_errors.log        # Logs if job crashes


# Activate your virtual environment (if using one)
eval "$(conda shell.bash hook)"
conda activate nlp

# Change to the directory containing your Python script
cd /users/home/thorirhh21/nlp/summary/finetune/

# Run your Python script
python sft_domar_train.py
