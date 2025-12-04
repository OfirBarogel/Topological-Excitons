# Topological-Excitons
Workflow:
1. create config and place in config folder
2. run on cluster, pass relative config file path to run.sh or config_paths.txt and run batch
3. download from cluster the datetime folder with 4 files: config, Exciton_Energys, F[0], Integrated_Fs
4. Analyze data with notebook
Additional:
If exploring parameter range write relative path to config_paths.txt and run batch script.
If I want to look at other bands use Fslicer on Fs.npy file by giving the path relative to tobiasholder-storage/Data to Slicer.sh as well as the index of desired band.
Notebook can read folders.txt (which contains all analyzed folders, analyze data to the correct folders and add those folders to folders.txt. 