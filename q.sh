#!/bin/bash
#SBATCH -J Topological_Excitons
#SBATCH -A tobiasholder-users_v2
#SBATCH -p tobiasholder-pool
#SBATCH --qos=owner                     
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
#mkdir -p logs

# --- pick up config path ---
CONFIG="${1:-${CONFIG_PATH:-}}"
if [[ -z "${CONFIG}" ]]; then
  echo "Usage: sbatch run.sh path/to/config.yml"
  echo "   or: CONFIG_PATH=path/to/config.yml sbatch run.sh"
  exit 1
fi
if [[ ! -f "$CONFIG" ]]; then
  echo "Config not found: $CONFIG"
  exit 2
fi
BASENAME="$(basename "$CONFIG" .yml)"

echo "Job $SLURM_JOB_ID on $(hostname) at $(date)"
echo "Config: $CONFIG"

module purge
module load python/python-anaconda3.6.5
python -m venv venv
export XDG_RUNTIME_DIR=/tmp/runtime-$USER
mkdir -p "$XDG_RUNTIME_DIR"; chmod 700 "$XDG_RUNTIME_DIR"

source venv/bin/activate
#python -m pip install -r requirements.txt
python 'Topological Excitons.py' --config "$CONFIG"
#python 'FSlicer.py' --config "$CONFIG"
STATUS=$?
deactivate

echo "Finished at $(date) with exit code $STATUS"
exit $STATUS
