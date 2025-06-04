import numpy as np
from pathlib import Path

if __name__ == "__main__":
    storage_path = Path("/tobiasholder-storage/ofirbarogel")
    data_folder = Path("Data/2025-06-03_17-13-53") #hard coded
    data_name = Path("Fs.npy")
    data_path = storage_path / data_folder / data_name
    f_matrix = np.load(data_path, mmap_mode='r')
    slice_0 = f_matrix[:, :, :, 0]  # This has shape (63, 63, 1024)
    data_name = Path("F[0].npy")
    data_path = storage_path / data_folder / data_name
    np.save(data_path,slice_0)