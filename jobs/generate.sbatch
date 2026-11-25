#!/bin/bash
#SBATCH --job-name=generate
#SBATCH --account=tml_2026
#SBATCH --partition=a100_normal_q
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/home/%u/tml/logs/%x-%j.out

# Full generation: all 2,400 rows (600 questions x clean/link/caps/two_copy) of Dominic's file.
# Resumable: if it stops, resubmit and it skips finished rows.
module reset
module load Miniforge3
source activate ~/.conda/envs/tml_a100
cd ~/tml
nvidia-smi --query-gpu=name,memory.total --format=csv
python src/generate.py --prompts ~/repo/data/prompts.jsonl --out outputs/generations.jsonl
