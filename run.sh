#!/bin/bash
#PBS -N Topological-Excitons
#PBS -q power-tobiasholder
#PBS -l nodes=1:ppn=96
#PBS -l walltime=48:00:00
#PBS -l mem=256gb
#PBS -m bea
#PBS -M ofirbarogel@mail.tau.ac.il
#PBS -j oe

# --- go to the submission directory ---
cd "$PBS_O_WORKDIR" || { echo "Failed to cd to \$PBS_O_WORKDIR"; exit 1; }

# --- get config path from argument or env ---
CONFIG="${1:-${CONFIG_PATH:-}}"
if [[ -z "$CONFIG" ]]; then
  echo "Usage: qsub TE.sh -F \"path/to/config.yml\""
  echo "   or: CONFIG_PATH=path/to/config.yml qsub TE.sh"
  exit 1
fi
if [[ ! -f "$CONFIG" ]]; then
  echo "Config not found: $CONFIG"
  exit 2
fi
BASENAME="$(basename "$CONFIG" .yml)"

# --- optional: own log file (PBS already joins stdout/err) ---
mkdir -p logs
LOGFILE="logs/job_${PBS_JOBID}_${BASENAME}.log"
exec > >(tee -a "$LOGFILE") 2>&1

echo "Job started on $(hostname) at $(date)"
echo "Working directory: $PWD"
echo "PBS Job ID: $PBS_JOBID"
echo "Config: $CONFIG"

# --- load Python & env ---
module load python/python-3.9.21

# Fix matplotlib backend issues (no DISPLAY)
export XDG_RUNTIME_DIR=/tmp/runtime-$USER
mkdir -p "$XDG_RUNTIME_DIR"
chmod 700 "$XDG_RUNTIME_DIR"

# Activate virtual environment
source te-env/bin/activate

# --- run your script with the config argument ---
python 'Topological Excitons.py' --config "$CONFIG"

STATUS=$?
echo "Job finished at $(date) with exit code $STATUS"

# Deactivate virtual environment
deactivate

exit $STATUS
