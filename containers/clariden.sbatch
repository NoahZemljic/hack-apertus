#!/usr/bin/env bash
# Submit from repository root: sbatch -A <project_account> containers/clariden.sbatch
#SBATCH --job-name=hack-apertus
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --time=00:15:00
#SBATCH --output=hack-apertus-%j.log
set -euo pipefail
srun --environment=./containers/clariden.edf.toml \
  bash /opt/hack-apertus/containers/start_pipeline.sh "$@"
