# Topological-Excitons
Workflow:
1. create config and place in config folder
2. run on cluster, pass relative config file path to run.sh or config_paths.txt and run batch
3. download from cluster the datetime folder with 4 files: config, Exciton_Energys, F[0], Integrated_Fs
4. Analyze data with notebook
Additional:
If multiple configs write relative paths to config_paths.txt and run batch script.
If I want to look at other bands use Fslicer on Fs.npy file by giving the path relative to tobiasholder-storage/Data to Slicer.sh as well as the index of desired band.
Notebook can read folders.txt (which contains all analyzed folders, analyze data to the correct folders and add those folders to folders.txt.

# Scripts:
Topological Excitons.ipynb: contains all the functions, workspace for code development and documentation. This script contains all the graphing.
Topological_Excitons.py: contains functions from the notebook relating to the solving of the BSE. main runs calculations from config.
ExcitonWF.py: contains functions from notebook relating to the exciton wavefunctions (anything relating to dot product)