#!/bin/bash
#SBATCH --job-name=dry-run
#SBATCH --account=tml_2026
#SBATCH --partition=a100_normal_q
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/home/%u/tml/logs/%x-%j.out

module reset
module load Miniforge3
source activate ~/.conda/envs/tml_a100
cd ~/tml
nvidia-smi --query-gpu=name,memory.total --format=csv
python src/dry_run.py --data data/nq/*.jsonl.gz --n 20 --out out/dry_run_$SLURM_JOB_ID
