#!/bin/bash

#SBATCH --job-name=Topological-Excitons
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=96
#SBATCH --time=48:00:00
#SBATCH --mem=500G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ofirbarogel@mail.tau.ac.il
# [ADDED] Optional: set a default/static array here, or pass --array at submission time (recommended)
# #SBATCH --array=0-0   # <-- override at submit time with: sbatch --array=0-(N-1) run.sh

# [ADDED] Resolve config list and select by array index
CONFIGS=(configs/*.yml)
if [[ ${#CONFIGS[@]} -eq 0 ]]; then
  echo "No config files found at configs/*.yml"; exit 2
fi

# If not submitted as an array, default to the first config (useful for local tests)
: "${SLURM_ARRAY_TASK_ID:=0}"  # [ADDED]
CFG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"  # [ADDED]
if [[ -z "$CFG" ]]; then
  echo "Array index $SLURM_ARRAY_TASK_ID is out of range (have ${#CONFIGS[@]} configs)"; exit 3  # [ADDED]
fi
BASENAME="$(basename "$CFG" .yml)"  # [ADDED]

# Go to the job's submission directory
cd "$SLURM_SUBMIT_DIR"

# Ensure logs directory exists
mkdir -p logs

# [CHANGED] Custom log file: include config name + array id for uniqueness
LOGFILE="logs/job_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${BASENAME}.log"
exec > >(tee -a "$LOGFILE") 2>&1

echo "Job started on $(hostname) at $(date)"
echo "Working directory: $PWD"
echo "Job ID: $SLURM_JOB_ID"
echo "Array task ID: $SLURM_ARRAY_TASK_ID"   # [ADDED]
echo "Config: $CFG"                           # [ADDED]

# Find and remove any .o<jobid> and .e<jobid> files in current directory
for ext in o e; do
    f=$(find . -maxdepth 1 -name "*.${ext}${SLURM_JOB_ID}" -print -quit)
    if [[ -n "$f" && -f "$f" ]]; then
        echo "Removing SLURM file: $f"
        rm "$f"
    fi
done

# Load Python module
module load python/python-3.9.21

# Fix matplotlib backend issues
export XDG_RUNTIME_DIR=/tmp/runtime-$USER
mkdir -p "$XDG_RUNTIME_DIR"
chmod 700 "$XDG_RUNTIME_DIR"

# Activate virtual environment
source te-env/bin/activate


# [CHANGED] Run the script with config path and outdir
python 'Topological Excitons.py' --config "$CFG"
echo "Job finished successfully at $(date)"

# Deactivate virtual environment
deactivate
