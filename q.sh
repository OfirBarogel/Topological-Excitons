#!/bin/bash

#SBATCH --job-name=Topological-Excitons
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=96
#SBATCH --time=48:00:00
#SBATCH --mem=256G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ofirbarogel@mail.tau.ac.il

# Go to the job's submission directory
cd "$SLURM_SUBMIT_DIR"

# Ensure logs directory exists
mkdir -p logs

# Custom log file
LOGFILE="logs/job_${SLURM_JOB_ID}.log"
exec > >(tee -a "$LOGFILE") 2>&1

echo "Job started on $(hostname) at $(date)"
echo "Working directory: $PWD"
echo "Job ID: $SLURM_JOB_ID"

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

# Run the script
python 'Topological Excitons.py'

echo "Job finished successfully at $(date)"

# Deactivate virtual environment
deactivate
