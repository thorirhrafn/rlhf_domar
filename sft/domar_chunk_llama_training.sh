#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=thorirhh21@ru.is        # for example uname@hi.is
#SBATCH --job-name=icellama
#SBATCH --nodes=2
#SBATCH --partition=gpu-2xA100
#SBATCH --time=06-04:00:00                   # run for 6 day maximum
#SBATCH --output=icellama_domar_chunk_v2_output.log
#SBATCH --error=icellama_domar_chunk_v2_errors.log        # Logs if job crashes


# Activate your virtual environment (if using one)
eval "$(conda shell.bash hook)"
conda activate nlp

# Change to the directory containing your Python script
cd /users/home/thorirhh21/nlp/summary/finetune/

# Run your Python script
python sft_domar_llama_chunk.py
