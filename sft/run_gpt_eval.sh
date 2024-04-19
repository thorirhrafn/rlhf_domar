#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=thorirhh21@ru.is        # for example uname@hi.is
#SBATCH --job-name=sft_eval
#SBATCH --nodes=1
#SBATCH --partition=gpu-1xA100
#SBATCH --time=04-04:00:00                   # run for 1 day maximum
#SBATCH --output=eval_job_output.log
#SBATCH --error=eval_job_errors.log        # Logs if job crashes


# Activate your virtual environment (if using one)
eval "$(conda shell.bash hook)"
conda activate nlp

# Change to the directory containing your Python script
cd /users/home/thorirhh21/nlp/summary/finetune/

# Run your Python script
python sft_eval_llama.py
