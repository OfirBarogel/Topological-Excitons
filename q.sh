#!/bin/bash

# PBS directives
#PBS -N Topological-Excitons
#PBS -l nodes=1:ppn=96 
#PBS -l walltime=24:00:00
#PBS -l mem=256gb
#PBS -m bea
#PBS -M ofirbarogel@mail.tau.ac.il
#PBS -j oe  # Join standard output and error

module load python/python-anaconda3.6.5

# Make sure that the node runs from the current directory
cd "$PBS_O_WORKDIR"

# Fix issue with matplotlib running in node
export XDG_RUNTIME_DIR=/tmp/runtime-$USER
mkdir -p $XDG_RUNTIME_DIR
chmod 700 $XDG_RUNTIME_DIR

# Set up python environement
python3.6 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt

python 'Topological Excitons.py' -Nk 12

deactivate
