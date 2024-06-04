#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=thorirhh21@ru.is        # for example uname@hi.is
#SBATCH --job-name=gpt_eval
#SBATCH --nodes=1
#SBATCH --partition=Jotunn-GPU
#SBATCH --time=01-00:00:00                   # run for 1 day maximum
#SBATCH --output=rouge_gpt1B_rlhf_output.log
#SBATCH --error=rouge_gpt1B_rlhf_errors.log        # Logs if job crashes


# Activate your virtual environment (if using one)
eval "$(conda shell.bash hook)"
conda activate nlp

# Change to the directory containing your Python script
cd /users/home/thorirhh21/nlp/summary/rogue/

# Run your Python script
python eval_rouge.py
