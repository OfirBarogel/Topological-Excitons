import numpy as np
from pathlib import Path
import argparse
import shutil
import re
import os
from numpy.lib.format import open_memmap


def StitchEvals(perq_dir, out_path, NQ):
    """
    Reads per-Q files and writes a memmap numpy array of shape (NQ, NQ, Nk2)
    out_path: path to output .dat (memmap) file
    """
    perq_dir = Path(perq_dir)

    # read Nk2 from any file
    with np.load(next(perq_dir.glob("Q_*_*.npz")), allow_pickle=True) as sample:
        Nk2 = int(sample["e_all"].shape[0])

    out_path = Path(out_path)
    # create memmap
    shape = (NQ, NQ, Nk2)
    dtype = np.float64
    print(f"Allocating memmap for evals_all: shape={shape}, dtype={dtype}")
    E = open_memmap(out_path, mode="w+", dtype=dtype, shape=shape)

    # fill
    for i in range(NQ):
        for j in range(NQ):
            pfile = perq_dir / f"Q_{i:03d}_{j:03d}.npz"
            if not pfile.exists():
                raise FileNotFoundError(f"Missing per-Q file: {pfile}")
            arr = np.load(pfile)
            E[i, j, :] = arr["e_all"]

    # flush to disk
    E.flush()
    print(f"Saved full eigenvalues to {out_path}")

# -------------------------
# Stitcher B: eigenvectors up to n_Fs (and their eigenvalues)
# -------------------------
def StitchEvecs(perq_dir, out_evec_path, NQ, n_Fs):
    """
    Writes:
      - out_evec_path memmap complex128 shape (NQ, NQ, Nk2, n_Fs)
    """
    perq_dir = Path(perq_dir)
    with np.load(next(perq_dir.glob("Q_*_*.npz")), allow_pickle=True) as sample:
        Nk2 = int(sample["e_all"].shape[0])
    # create memmaps
    shape_vecs = (NQ, NQ, Nk2, n_Fs)
    print(f"Allocating memmap for evecs_keep: shape={shape_vecs}")
    F = open_memmap(out_evec_path, mode="w+", dtype=np.complex128, shape=shape_vecs)
    for i in range(NQ):
        for j in range(NQ):
            pfile = perq_dir / f"Q_{i:03d}_{j:03d}.npz"
            if not pfile.exists():
                raise FileNotFoundError(f"Missing per-Q file: {pfile}")
            arr = np.load(pfile)
            f_keep = arr["f_keep"]  # (Nk2, n_keep) where n_keep may be >= n_Fs
            n_keep = f_keep.shape[1]
            if n_keep < n_Fs:
                raise ValueError(f"per-Q file {pfile} has only {n_keep} vectors but n_Fs={n_Fs}")
            # store only requested n_Fs
            F[i, j, :, :] = f_keep[:, :n_Fs].reshape(Nk2, n_Fs)

    F.flush()
    print(f"Saved eigenvectors (first {n_Fs}) to {out_evec_path}")

def CopyDirectory(src, dst):
    os.makedirs(dst, exist_ok=True)

    for name in os.listdir(src):
        src_path = os.path.join(src, name)
        dst_path = os.path.join(dst, name)

        if os.path.isdir(src_path):
            CopyDirectory(src_path, dst_path)
        else:
            shutil.copy2(src_path, dst_path)

from pathlib import Path
import shutil


def CopyDownloadSubset(download_dir, band):
    source = Path(download_dir).resolve()

    if not source.is_dir():
        raise NotADirectoryError(str(source))

    destination = source / f"{source.name}_Subset"

    # Exclude current and previously generated output folders.
    excluded = {
        f"{source.name}_Subset",
    }

    children = [
        folder for folder in source.iterdir()
        if folder.is_dir() and folder.name not in excluded
    ]

    destination.mkdir(exist_ok=True)
    copied = 0

    for folder in sorted(children):
        config = folder / "config.yml"
        energies = folder / "Exciton_Energys.npy"

        # Literal matching avoids glob interpreting [0] as a pattern.
        bc_files = [
            file for file in folder.iterdir()
            if file.is_file()
            and file.name.startswith((f"Vprod_U[{band}]BC_C_", f"U[{band}]BC_C_"))
            and file.suffix == ".npy"
        ]

        missing = []
        if not config.is_file():
            missing.append("config.yml")
        if not energies.is_file():
            missing.append("Exciton_Energys.npy")
        if not bc_files:
            missing.append(f"Vprod_U[{band}]BC_C_*.npy or U[{band}]BC_C_*")

        if missing:
            print(f"Skipping {folder.name}: missing {', '.join(missing)}")
            continue

        target = destination / folder.name
        target.mkdir(exist_ok=True)

        for file in [config, energies] + bc_files:
            shutil.copy2(file, target / file.name)

        copied += 1

    print(f"Copied {copied} folders to {destination}")
    return destination

def CopyLatestDirectories(
    folder_list_path,
    storage_path,
    down_dir
):
    data_root = storage_path/"Data"
    download_root = storage_path/"Download"/down_dir

    download_root.mkdir(parents=True, exist_ok=True)

    with folder_list_path.open("r") as file:
        requested_names = [
            line.strip()
            for line in file
            if line.strip()
        ]

    for base_name in requested_names:
        pattern = re.compile(
            r"^{}-slurm(?P<job_id>\d+)-s(?P<step>\d+)$".format(
                re.escape(base_name)
            )
        )

        matches = []

        for path in data_root.iterdir():
            if not path.is_dir():
                continue

            match = pattern.fullmatch(path.name)

            if match is not None:
                job_id = int(match.group("job_id"))
                step = int(match.group("step"))
                matches.append((job_id, step, path))

        if not matches:
            print("NOT FOUND: {}".format(base_name))
            continue

        # Highest job ID wins. Step breaks a tie if necessary.
        job_id, step, job_directory = max(
            matches,
            key=lambda item: (item[0], item[1]),
        )

        source_path = job_directory / base_name
        destination_path = download_root / base_name

        print(
            "Copying job {}: {} -> {}".format(
                job_id,
                source_path,
                destination_path,
            )
        )

        CopyDirectory(
            source_path,
            destination_path,
        )

def MainFSlice():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="Input directory")
    ap.add_argument("--band", required=True, help="band index to slice")
    args = ap.parse_args()
    storage_path = Path("/tobiasholder-storage/ofirbarogel")
    dir_name = args.dir
    band_num = args.band
    data_folder = Path(f"Data/{dir_name}")
    data_name = Path("Fs.npy")
    data_path = storage_path / data_folder / data_name
    f_matrix = np.load(data_path, mmap_mode='r')
    slice_0 = f_matrix[:, :, :, band_num]  # This has shape (63, 63, 1024) 
    data_name = Path(f"F[{band_num}].npy")
    data_path = storage_path / data_folder / data_name
    np.save(data_path,slice_0)

def MainMinDownload():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="Input directory")
    ap.add_argument("--band", required=True, help="band index to slice")
    args = ap.parse_args()
    storage_path = Path("/tobiasholder-storage/ofirbarogel")
    dir_name = args.dir
    band = args.band
    down_folder = Path(f"{storage_path}/Download/{dir_name}")
    CopyDownloadSubset(down_folder,band)

def MainCopyToDownDir():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="Download dir")
    ap.add_argument("--band", required=True, help="band index to slice")
    args = ap.parse_args()
    storage_path = Path("/tobiasholder-storage/ofirbarogel")
    dir_name = Path(args.dir)
    band_num = args.band
    folder_list_path = Path("copy_folders.txt")
    CopyLatestDirectories(folder_list_path, storage_path, dir_name)
    
if __name__ == "__main__":
    MainCopyToDownDir()