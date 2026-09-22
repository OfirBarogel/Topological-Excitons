# Disk-backed exciton geometry pipeline

Keep these three files in the same directory:

- `Topological_Excitons.py`
- `ExcitonWF.py`
- `DataManager.py`

The filenames matter because the modules import one another by these names.

## 1. Run the complete calculation

```bash
python Topological_Excitons.py \
  --config /path/to/config.yml \
  --n_workers 16
```

This runs the noninteracting problem, the parallel BSE, the legacy `F[n]` and
Vprod diagnostics, and the disk-backed projector/link geometry for every
positive `n_Fs` band.

To avoid stitching and saving `F[n]` and to skip the legacy Vprod diagnostic:

```bash
python Topological_Excitons.py \
  --config /path/to/config.yml \
  --n_workers 16 \
  --geometry_only
```

The final geometry still runs in `--geometry_only` mode because it reads each
band directly from the per-Q BSE files.

## 2. Complete or resume geometry later with ExcitonWF.py

Point `--dir` to the result directory that directly contains `config.yml`:

```bash
python ExcitonWF.py \
  --dir /absolute/path/to/job/result_name \
  --band 0 \
  --n_workers 16
```

Omit `--band` to complete every state stored in `f_keep`:

```bash
python ExcitonWF.py \
  --dir /absolute/path/to/job/result_name \
  --n_workers 16
```

For a cluster-relative path, `--storage_path` is the prefix. Its default is
`/tobiasholder-storage/ofirbarogel`, so this is also valid:

```bash
python ExcitonWF.py \
  --dir Data/config-slurm123/config \
  --band 0 \
  --n_workers 16
```

The expected layout is:

```text
job/
  inputs/
    ni_wf.npy
    phase_data.npz
  perQ/
    Q_000_000.npz
    ...
  ExcitonStates_U[0]/
  ExcitonGeometry_U[0]_per_Q/
  result_name/
    config.yml
    ExcitonGeometry_U[0].npz
```

`ExcitonWF.py` can resume a partially completed state or geometry cache. Use
`--no_resume` only when a forced rebuild is wanted.

## 3. Final output

For band `n`, the final file is:

```text
result_name/ExcitonGeometry_U[n].npz
```

Load it as follows:

```python
import numpy as np

with np.load("ExcitonGeometry_U[0].npz", allow_pickle=False) as result:
    BC_projector = result["BC_projector"]  # curvature density on the Q grid
    BC_links = result["BC_links"]          # plaquette Berry flux
    g_11 = result["g_11"]
    g_12 = result["g_12"]
    g_22 = result["g_22"]
    C_projector = float(result["C_projector"])
    C_links = float(result["C_links"])
```

`BC_projector` is not pre-multiplied by the grid-cell factor. The assembler
uses `sum(BC_projector) * dQ1 * dQ2 / (2*pi)` for `C_projector`.
`BC_links` is already a flux per plaquette, so `C_links` is its direct sum
divided by `2*pi`.

The physical steps `dQ1` and `dQ2` are derived from `cfg["Q_frac"]`. This is
important because `Grid.Handle: extend` has twice the fractional spacing of
`Grid.Handle: sew` at the same `NQ`.

## 4. Resume rules

- A BSE per-Q file is reused only when it is readable, has the expected array
  shapes, and carries the current BSE configuration signature.
- A cached exciton state is reused only when its BSE inputs, band selection,
  single-particle inputs, normalization choice, and Q-range convention match.
- A geometry point is reused only when its state signature, grid steps, and
  overlap tolerance match.
- The final full-grid `.npz` is assembled again after the per-Q checks; this is
  inexpensive and guarantees a complete final result.

Older per-Q files without `bse_signature` can still be consumed directly by
`ExcitonWF.py`. Running them through the corrected `Topological_Excitons.py`
will rebuild them once so future BSE resumes are configuration-safe.

## 5. Boundary convention to verify

The geometry and link workers identify opposite Q-grid edges with a simple
modulo wrap. This is the intended convention for the doubled, `L=0`
`Grid.Handle: extend` torus. If `Grid.Handle: sew` requires a nontrivial Q-seam
relabeling or sewing transformation in your formulation, that transformation
is not implemented here; seam plaquettes should not be interpreted until it
is supplied.
