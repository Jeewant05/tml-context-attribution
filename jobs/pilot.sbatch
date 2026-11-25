#!/bin/bash
#SBATCH --job-name=pilot
#SBATCH --account=tml_2026
#SBATCH --partition=a100_normal_q
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=00:50:00
#SBATCH --output=/home/%u/tml/logs/%x-%j.out

# Pilot: 50 new questions (seed 1, not the dry-run sample), three link wordings side by side,
# all caps, and two copies of each link wording. Same planted positions across wordings.
module reset
module load Miniforge3
source activate ~/.conda/envs/tml_a100
cd ~/tml
nvidia-smi --query-gpu=name,memory.total --format=csv
python src/dry_run.py --data data/nq/*.jsonl.gz --n 50 --seed 1 \
  --conditions link,link_both,link_override,caps,two_copy,two_copy_both,two_copy_override \
  --out out/pilot_$SLURM_JOB_ID
