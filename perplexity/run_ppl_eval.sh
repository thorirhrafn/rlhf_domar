#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=thorirhh21@ru.is        # for example uname@hi.is
#SBATCH --job-name=ppl_eval
#SBATCH --nodes=1
#SBATCH --partition=gpu-2xA100
#SBATCH --time=00-12:00:00                   # run for 12 hours maximum
#SBATCH --output=ppl_llama_sft_output.log
#SBATCH --error=ppl_llama_sft_errors.log        # Logs if job crashes


# Activate your virtual environment (if using one)
eval "$(conda shell.bash hook)"
conda activate nlp

# Change to the directory containing your Python script
cd /users/home/thorirhh21/nlp/summary/perplexity/

# Run your Python script
python eval_perplexity.py
