#!/bin/bash
#SBATCH -J Topological_Excitons
#SBATCH -A tobiasholder-users_v2
#SBATCH -p power-general-shared-pool
#SBATCH --qos=public
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

# Temp dirs away from /tmp
mkdir -p "$HOME/tmp"
export TMPDIR="$HOME/tmp"
export TMP="$HOME/tmp"
export TEMP="$HOME/tmp"
export XDG_RUNTIME_DIR="$HOME/tmp/runtime-$USER"
mkdir -p "$XDG_RUNTIME_DIR"
chmod 700 "$XDG_RUNTIME_DIR"

# --- pick up config path ---
CONFIG="${1:-${CONFIG_PATH:-}}"
DOWN_FOLDER="${2:-}"
if [[ -z "${CONFIG}" ]]; then
  echo "Usage: sbatch run.sh path/to/config.yml"
  echo "   or: CONFIG_PATH=path/to/config.yml sbatch run.sh"
  exit 1
fi
if [[ ! -f "$CONFIG" ]]; then
  echo "Config not found: $CONFIG"
  exit 2
fi

echo "Job $SLURM_JOB_ID on $(hostname) at $(date)"
echo "Config: $CONFIG"
echo "Download folder: $DOWN_FOLDER"

module purge
module load python-anaconda3-2025.12

[[ -d venv ]] || python -m venv venv
source venv/bin/activate

if [[ -n "$DOWN_FOLDER" ]]; then
    python Topological_Excitons.py \
        --config "$CONFIG" \
        --down_folder "$DOWN_FOLDER"
else
    python Topological_Excitons.py \
        --config "$CONFIG"
fi

STATUS=$?

deactivate
echo "Finished at $(date) with exit code $STATUS"
exit $STATUS
