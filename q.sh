#!/bin/bash

# PBS directives
#PBS -N Topological-Excitons
#PBS -l nodes=1:ppn=96 
#PBS -l walltime=24:00:00
#PBS -l mem=256gb
#PBS -m bea
#PBS -M ofirbarogel@mail.tau.ac.il
#PBS -j oe

# Go to job's submission directory
cd "$PBS_O_WORKDIR"

# Load working Python module
module load python/python-3.9.21

# Fix matplotlib backend issues on headless nodes
export XDG_RUNTIME_DIR=/tmp/runtime-$USER
mkdir -p $XDG_RUNTIME_DIR
chmod 700 $XDG_RUNTIME_DIR

# Activate existing venv (must already be created)
source te-env/bin/activate

# Run script
python 'Topological Excitons.py' -Nk 12

# Deactivate
deactivate
