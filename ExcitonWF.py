import numpy as np
from pathlib import Path
import argparse
import Topological_Excitons as TE
import gc
import DataManager
import hashlib
import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

STATE_FORMAT_VERSION = 1
GEOMETRY_FORMAT_VERSION = 1


def _GridShape(cfg):
    """Return (NQ1, NQ2); current square-grid configs use cfg['NQ']."""
    return (
        int(cfg.get("NQ1", cfg["NQ"])),
        int(cfg.get("NQ2", cfg["NQ"])),
    )


def _BSEQPath(folder, q1, q2):
    return Path(folder) / "Q_{0:03d}_{1:03d}.npz".format(q1, q2)


def _StateQPath(folder, q1, q2):
    return Path(folder) / "X_Q_{0:03d}_{1:03d}.npz".format(q1, q2)


def _GeometryQPath(folder, q1, q2):
    return Path(folder) / "geometry_Q_{0:03d}_{1:03d}.npz".format(q1, q2)


def _FileStamp(path):
    """Cheap source fingerprint used to invalidate stale cached results."""
    path = Path(path).resolve()
    if not path.is_file():
        raise IOError("Required input file does not exist: {0}".format(path))
    stat = path.stat()
    return "{0}|{1}|{2}".format(path, stat.st_size, stat.st_mtime_ns)


def _StateCacheSignature(
    cfg,
    bse_dir,
    ni_wf_path,
    phase_data_path,
    band_c,
    band_v,
    exciton_state_index,
    normalize,
    range_handle,
):
    """Hash all settings and source-file timestamps relevant to X(Q)."""
    nq1, nq2 = _GridShape(cfg)
    payload = {
        "format_version": STATE_FORMAT_VERSION,
        "NQ1": nq1,
        "NQ2": nq2,
        "Nk": int(cfg["Nk"]),
        "band_c": int(band_c),
        "band_v": int(band_v),
        "exciton_state_index": int(exciton_state_index),
        "normalize": bool(normalize),
        "range_handle": str(range_handle),
        "ni_wf": _FileStamp(ni_wf_path),
        "phase_data": _FileStamp(phase_data_path),
    }

    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    )
    for q1 in range(nq1):
        for q2 in range(nq2):
            digest.update(_FileStamp(_BSEQPath(bse_dir, q1, q2)).encode("utf-8"))
    return digest.hexdigest(), payload


def _GeometrySignature(state_signature, dQ1, dQ2, overlap_tolerance):
    payload = {
        "format_version": GEOMETRY_FORMAT_VERSION,
        "state_signature": str(state_signature),
        "dQ1": float(dQ1),
        "dQ2": float(dQ2),
        "overlap_tolerance": float(overlap_tolerance),
    }
    signature = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return signature, payload


def _WriteManifest(path, signature, settings):
    """Write a small human-readable cache manifest atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    content = dict(settings)
    content["signature"] = str(signature)
    with tmp.open("w") as handle:
        json.dump(content, handle, indent=2, sort_keys=True)
    os.replace(str(tmp), str(path))


def _ReadManifest(path):
    path = Path(path)
    if not path.is_file():
        raise IOError("Missing cache manifest: {0}".format(path))
    with path.open("r") as handle:
        return json.load(handle)


def _ResolveWorkerCount(cfg, n_workers):
    if n_workers is not None:
        return max(1, int(n_workers))

    runtype = cfg.get("RunType", "Local")
    if runtype == "Local":
        return max(1, multiprocessing.cpu_count() - 1)
    if runtype == "Cluster":
        return max(
            1,
            int(os.environ.get("SLURM_CPUS_PER_TASK", multiprocessing.cpu_count())),
        )
    raise ValueError("Unknown RunType: {0}".format(runtype))


def _PrintProgress(stage, completed, total):
    """Print at most about twenty progress messages per stage."""
    stride = max(1, total // 20)
    if completed == total or completed % stride == 0:
        print("{0}: {1}/{2} complete".format(stage, completed, total))


def _LoadBSEState(bse_dir, q1, q2, nk, exciton_state_index):
    """Load F_n(p,Q) from one BSE per-Q file and reshape it to (Nk, Nk)."""
    path = _BSEQPath(bse_dir, q1, q2)
    with np.load(str(path), allow_pickle=False) as data:
        if "f_keep" not in data:
            raise KeyError("{0} does not contain 'f_keep'".format(path))
        f_keep = data["f_keep"]

    nk2 = nk * nk
    if f_keep.ndim == 2 and f_keep.shape[0] == nk2:
        n_keep = f_keep.shape[1]
        if not -n_keep <= exciton_state_index < n_keep:
            raise IndexError(
                "Exciton band {0} is outside the {1} states in {2}".format(
                    exciton_state_index, n_keep, path
                )
            )
        return np.asarray(
            f_keep[:, exciton_state_index], dtype=np.complex128
        ).reshape(nk, nk)

    if f_keep.ndim == 3 and f_keep.shape[:2] == (nk, nk):
        n_keep = f_keep.shape[2]
        if not -n_keep <= exciton_state_index < n_keep:
            raise IndexError(
                "Exciton band {0} is outside the {1} states in {2}".format(
                    exciton_state_index, n_keep, path
                )
            )
        return np.asarray(
            f_keep[:, :, exciton_state_index], dtype=np.complex128
        )

    raise ValueError(
        "Unsupported f_keep shape {0} in {1}; expected "
        "(Nk*Nk,n_keep) or (Nk,Nk,n_keep)".format(f_keep.shape, path)
    )


def BuildExcitonStateOneQDisk(
    Fn_Q,
    ni_wf,
    phase_data,
    cfg,
    band_c,
    band_v,
    Q1,
    Q2,
    range_handle="forward",
):
    """Construct one flattened exciton state X(Q) from F_n(p,Q).

    The returned vector uses the composite basis (p1, p2, orbital_c,
    orbital_v).  No projector and no other Q point are constructed here.
    """
    Fn_Q = np.asarray(Fn_Q)
    if Fn_Q.ndim != 2:
        raise ValueError("Fn_Q must have shape (Np1, Np2)")

    np1, np2 = Fn_Q.shape
    nk = int(cfg["Nk"])
    if (np1, np2) != (nk, nk):
        raise ValueError(
            "Fn_Q has shape {0}, but cfg['Nk'] implies ({1},{1})".format(
                Fn_Q.shape, nk
            )
        )

    if range_handle == "forward":
        q1i, q2i = Q1, Q2
    elif range_handle == "center":
        nq1, nq2 = _GridShape(cfg)
        q1i = Q1 - nq1 // 2
        q2i = Q2 - nq2 // 2
    else:
        raise ValueError(
            "range_handle must be 'forward' or 'center', got {0}".format(
                range_handle
            )
        )

    norb = ni_wf.shape[2]
    state = np.empty((np1, np2, norb * norb), dtype=np.complex128)

    for p1 in range(np1):
        for p2 in range(np2):
            kp = (p1, p2, 0, 0)
            kv = TE.PBCAddInd(kp, (-q1i, -q2i), nk)
            kc = TE.PBCAddInd(kp, (q1i, q2i), nk)

            uc = TE.u_ext(ni_wf, phase_data, kc, band_c)
            uv = TE.u_ext(ni_wf, phase_data, kv, band_v)

            # Component (a,b) is F_n(p,Q) u_c,a(p+Q/2) u_v,b*(p-Q/2).
            state[p1, p2, :] = (
                Fn_Q[p1, p2] * np.kron(uc, np.conj(uv))
            )

    return state.reshape(-1)


def _SaveExcitonState(path, state, signature, q1, q2):
    """Save one state and its cache signature atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(
        str(tmp),
        X=np.asarray(state, dtype=np.complex128),
        signature=np.asarray(str(signature)),
        Q1=np.int64(q1),
        Q2=np.int64(q2),
    )
    os.replace(str(tmp), str(path))


def _StateFileMatches(path, signature):
    path = Path(path)
    if not path.is_file():
        return False
    try:
        with np.load(str(path), allow_pickle=False) as data:
            return (
                "X" in data
                and "signature" in data
                and str(data["signature"]) == str(signature)
            )
    except (IOError, OSError, ValueError, KeyError):
        return False


def _LoadCachedExcitonState(state_dir, q1, q2, expected_signature):
    path = _StateQPath(state_dir, q1, q2)
    with np.load(str(path), allow_pickle=False) as data:
        if str(data["signature"]) != str(expected_signature):
            raise ValueError("Incompatible cached exciton state: {0}".format(path))
        state = np.asarray(data["X"], dtype=np.complex128)
    if state.ndim != 1:
        raise ValueError("Cached state must be one-dimensional: {0}".format(path))
    return state


def BuildExcitonStateOneQTask(
    Q1,
    Q2,
    cfg,
    bse_dir,
    ni_wf_path,
    phase_data_path,
    state_dir,
    band_c,
    band_v,
    exciton_state_index,
    normalize,
    range_handle,
    state_signature,
):
    """Parallel worker that owns and saves exactly one X(Q)."""
    nk = int(cfg["Nk"])
    Fn_Q = _LoadBSEState(bse_dir, Q1, Q2, nk, exciton_state_index)
    ni_wf = np.load(str(ni_wf_path), mmap_mode="r")
    phase_data = np.load(str(phase_data_path), allow_pickle=False)
    try:
        state = BuildExcitonStateOneQDisk(
            Fn_Q,
            ni_wf,
            phase_data,
            cfg,
            band_c,
            band_v,
            Q1,
            Q2,
            range_handle=range_handle,
        )
    finally:
        close = getattr(phase_data, "close", None)
        if close is not None:
            close()

    if normalize:
        norm = np.linalg.norm(state)
        if not np.isfinite(norm) or norm == 0.0:
            raise ValueError(
                "Cannot normalize exciton state at Q=({0},{1}); norm={2}".format(
                    Q1, Q2, norm
                )
            )
        state = state / norm

    out = _StateQPath(state_dir, Q1, Q2)
    _SaveExcitonState(out, state, state_signature, Q1, Q2)
    return str(out)


def RunExcitonStateCacheParallel(
    cfg,
    bse_dir,
    ni_wf_path,
    phase_data_path,
    state_dir,
    band_c,
    band_v,
    exciton_state_index=0,
    normalize=True,
    range_handle="forward",
    n_workers=None,
    resume=True,
):
    """Build or resume one cached exciton state for every Q point."""
    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    nq1, nq2 = _GridShape(cfg)

    signature, settings = _StateCacheSignature(
        cfg,
        bse_dir,
        ni_wf_path,
        phase_data_path,
        band_c,
        band_v,
        exciton_state_index,
        normalize,
        range_handle,
    )
    _WriteManifest(state_dir / "state_cache_manifest.json", signature, settings)

    tasks = []
    for q1 in range(nq1):
        for q2 in range(nq2):
            path = _StateQPath(state_dir, q1, q2)
            if resume and _StateFileMatches(path, signature):
                continue
            tasks.append((q1, q2))

    workers = _ResolveWorkerCount(cfg, n_workers)
    if tasks:
        print(
            "Building {0} exciton states with {1} workers...".format(
                len(tasks), workers
            )
        )
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    BuildExcitonStateOneQTask,
                    q1,
                    q2,
                    cfg,
                    str(bse_dir),
                    str(ni_wf_path),
                    str(phase_data_path),
                    str(state_dir),
                    int(band_c),
                    int(band_v),
                    int(exciton_state_index),
                    bool(normalize),
                    str(range_handle),
                    signature,
                )
                for q1, q2 in tasks
            ]
            for completed, future in enumerate(as_completed(futures), 1):
                future.result()
                _PrintProgress("State cache", completed, len(tasks))
    else:
        print("All compatible exciton-state files already exist.")

    return str(state_dir), signature


def GeometryFromFiveStates(x0, x1p, x1m, x2p, x2m, dQ1, dQ2):
    """Calculate projector BC and metric using state overlaps only."""
    dQ1 = float(dQ1)
    dQ2 = float(dQ2)
    if dQ1 == 0.0 or dQ2 == 0.0:
        raise ValueError("dQ1 and dQ2 must be nonzero")

    def trace2(x, y):
        overlap = np.vdot(x, y)
        return float(np.real(overlap * np.conj(overlap)))

    def trace3(x, y, z):
        return np.vdot(x, y) * np.vdot(y, z) * np.vdot(z, x)

    d1 = ((x1p, 1.0), (x1m, -1.0))
    d2 = ((x2p, 1.0), (x2m, -1.0))

    tr_d1_d1 = sum(
        sa * sb * trace2(xa, xb)
        for xa, sa in d1
        for xb, sb in d1
    ) / (4.0 * dQ1 * dQ1)

    tr_d2_d2 = sum(
        sa * sb * trace2(xa, xb)
        for xa, sa in d2
        for xb, sb in d2
    ) / (4.0 * dQ2 * dQ2)

    tr_d1_d2 = 0.0
    tr_p_d1_d2 = 0.0j
    tr_p_d2_d1 = 0.0j
    for x1, s1 in d1:
        for x2, s2 in d2:
            sign = s1 * s2
            tr_d1_d2 += sign * trace2(x1, x2)
            tr_p_d1_d2 += sign * trace3(x0, x1, x2)
            tr_p_d2_d1 += sign * trace3(x0, x2, x1)

    tr_d1_d2 /= 4.0 * dQ1 * dQ2
    tr_p_d1_d2 /= 4.0 * dQ1 * dQ2
    tr_p_d2_d1 /= 4.0 * dQ1 * dQ2

    BC_projector = np.real(1j * (tr_p_d1_d2 - tr_p_d2_d1))
    g_11 = 0.5 * np.real(tr_d1_d1)
    g_12 = 0.5 * np.real(tr_d1_d2)
    g_22 = 0.5 * np.real(tr_d2_d2)
    return (
        float(BC_projector),
        float(g_11),
        float(g_12),
        float(g_22),
    )


def _UnitLink(left_state, center_state, tolerance, label):
    overlap = np.vdot(left_state, center_state)
    magnitude = np.abs(overlap)
    if not np.isfinite(magnitude) or magnitude <= tolerance:
        raise ValueError(
            "Cannot define {0}; overlap magnitude is {1}".format(label, magnitude)
        )
    return overlap / magnitude


def _SaveGeometryPoint(
    path,
    BC_projector,
    g_11,
    g_12,
    g_22,
    link_1,
    link_2,
    signature,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(
        str(tmp),
        BC_projector=np.float64(BC_projector),
        g_11=np.float64(g_11),
        g_12=np.float64(g_12),
        g_22=np.float64(g_22),
        link_1=np.complex128(link_1),
        link_2=np.complex128(link_2),
        signature=np.asarray(str(signature)),
    )
    os.replace(str(tmp), str(path))


def _GeometryFileMatches(path, signature):
    path = Path(path)
    if not path.is_file():
        return False
    required = (
        "BC_projector",
        "g_11",
        "g_12",
        "g_22",
        "link_1",
        "link_2",
        "signature",
    )
    try:
        with np.load(str(path), allow_pickle=False) as data:
            return (
                all(key in data for key in required)
                and str(data["signature"]) == str(signature)
            )
    except (IOError, OSError, ValueError, KeyError):
        return False


def ExcitonGeometryOneQTask(
    Q1,
    Q2,
    cfg,
    state_dir,
    geometry_dir,
    state_signature,
    geometry_signature,
    dQ1,
    dQ2,
    overlap_tolerance,
):
    """Load five cached states; save projector geometry and two links."""
    nq1, nq2 = _GridShape(cfg)
    coords = {
        "0": (Q1, Q2),
        "1p": ((Q1 + 1) % nq1, Q2),
        "1m": ((Q1 - 1) % nq1, Q2),
        "2p": (Q1, (Q2 + 1) % nq2),
        "2m": (Q1, (Q2 - 1) % nq2),
    }
    states = {
        label: _LoadCachedExcitonState(state_dir, q1, q2, state_signature)
        for label, (q1, q2) in coords.items()
    }

    BC_projector, g_11, g_12, g_22 = GeometryFromFiveStates(
        states["0"],
        states["1p"],
        states["1m"],
        states["2p"],
        states["2m"],
        dQ1,
        dQ2,
    )

    # Same orientation used in GetBerryCurvature1band:
    # link_1(Q) = <X(Q-e1)|X(Q)> / |...|, and similarly for link_2.
    link_1 = _UnitLink(
        states["1m"],
        states["0"],
        overlap_tolerance,
        "Q1 link at Q=({0},{1})".format(Q1, Q2),
    )
    link_2 = _UnitLink(
        states["2m"],
        states["0"],
        overlap_tolerance,
        "Q2 link at Q=({0},{1})".format(Q1, Q2),
    )

    out = _GeometryQPath(geometry_dir, Q1, Q2)
    _SaveGeometryPoint(
        out,
        BC_projector,
        g_11,
        g_12,
        g_22,
        link_1,
        link_2,
        geometry_signature,
    )
    return str(out)


def AssembleExcitonGeometry(
    cfg,
    geometry_dir,
    output_path,
    geometry_signature,
    dQ1,
    dQ2,
):
    """Assemble full-grid projector geometry and link-method curvature."""
    nq1, nq2 = _GridShape(cfg)
    shape = (nq1, nq2)
    BC_projector = np.empty(shape, dtype=float)
    g_11 = np.empty(shape, dtype=float)
    g_12 = np.empty(shape, dtype=float)
    g_22 = np.empty(shape, dtype=float)
    link_1 = np.empty(shape, dtype=np.complex128)
    link_2 = np.empty(shape, dtype=np.complex128)

    for q1 in range(nq1):
        for q2 in range(nq2):
            path = _GeometryQPath(geometry_dir, q1, q2)
            if not _GeometryFileMatches(path, geometry_signature):
                raise IOError(
                    "Missing or incompatible geometry result: {0}".format(path)
                )
            with np.load(str(path), allow_pickle=False) as data:
                BC_projector[q1, q2] = data["BC_projector"]
                g_11[q1, q2] = data["g_11"]
                g_12[q1, q2] = data["g_12"]
                g_22[q1, q2] = data["g_22"]
                link_1[q1, q2] = data["link_1"]
                link_2[q1, q2] = data["link_2"]

    # Exact plaquette orientation from the previous GetBerryCurvature1band.
    plaquette = (
        link_1
        * np.roll(link_2, 1, axis=0)
        * np.conj(np.roll(link_1, 1, axis=1))
        * np.conj(link_2)
    )
    BC_links = np.angle(plaquette)

    b1 = np.asarray(cfg["rec_vec"][0], dtype=float)
    b2 = np.asarray(cfg["rec_vec"][1], dtype=float)
    cell_area = abs(b1[0] * b2[1] - b1[1] * b2[0])
    dA = cell_area / float(nq1 * nq2)
    BC_projector_flux = BC_projector * dQ1 * dQ2
    C_projector = np.sum(BC_projector_flux) / (2.0 * np.pi)
    C_links = np.sum(BC_links) / (2.0 * np.pi)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix != ".npz":
        output_path = output_path.with_suffix(".npz")
    tmp = output_path.with_name(output_path.stem + ".tmp.npz")
    np.savez_compressed(
        str(tmp),
        #BC_projector=BC_projector,
        BC_projector=BC_projector_flux,
        BC_links=BC_links,
        g_11=g_11,
        g_12=g_12,
        g_22=g_22,
        C_projector=np.float64(C_projector),
        C_links=np.float64(C_links),
    )
    os.replace(str(tmp), str(output_path))
    return str(output_path)


def RunExcitonGeometryFromCacheParallel(
    cfg,
    state_dir,
    geometry_dir,
    output_path,
    dQ1,
    dQ2,
    n_workers=None,
    resume=True,
    overlap_tolerance=1.0e-14,
):
    """Calculate/resume geometry and links from a completed state cache."""
    state_dir = Path(state_dir)
    geometry_dir = Path(geometry_dir)
    geometry_dir.mkdir(parents=True, exist_ok=True)
    nq1, nq2 = _GridShape(cfg)

    state_manifest = _ReadManifest(state_dir / "state_cache_manifest.json")
    state_signature = state_manifest["signature"]
    for q1 in range(nq1):
        for q2 in range(nq2):
            path = _StateQPath(state_dir, q1, q2)
            if not _StateFileMatches(path, state_signature):
                raise IOError(
                    "State cache is incomplete or incompatible at Q=({0},{1}): {2}".format(
                        q1, q2, path
                    )
                )

    geometry_signature, settings = _GeometrySignature(
        state_signature, dQ1, dQ2, overlap_tolerance
    )
    _WriteManifest(
        geometry_dir / "geometry_manifest.json", geometry_signature, settings
    )

    tasks = []
    for q1 in range(nq1):
        for q2 in range(nq2):
            path = _GeometryQPath(geometry_dir, q1, q2)
            if resume and _GeometryFileMatches(path, geometry_signature):
                continue
            tasks.append((q1, q2))

    workers = _ResolveWorkerCount(cfg, n_workers)
    if tasks:
        print(
            "Calculating {0} geometry points with {1} workers...".format(
                len(tasks), workers
            )
        )
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    ExcitonGeometryOneQTask,
                    q1,
                    q2,
                    cfg,
                    str(state_dir),
                    str(geometry_dir),
                    state_signature,
                    geometry_signature,
                    float(dQ1),
                    float(dQ2),
                    float(overlap_tolerance),
                )
                for q1, q2 in tasks
            ]
            for completed, future in enumerate(as_completed(futures), 1):
                future.result()
                _PrintProgress("Geometry", completed, len(tasks))
    else:
        print("All compatible geometry-point files already exist.")

    return AssembleExcitonGeometry(
        cfg,
        geometry_dir,
        output_path,
        geometry_signature,
        dQ1,
        dQ2,
    )


def RunExcitonGeometryPipeline(
    cfg,
    bse_dir,
    ni_wf_path,
    phase_data_path,
    state_dir,
    geometry_dir,
    output_path,
    band_c,
    band_v,
    exciton_state_index=0,
    dQ1=1.0, #|b1|/NQ
    dQ2=1.0,
    normalize=True,
    range_handle="forward",
    n_workers=None,
    resume=True,
    overlap_tolerance=1.0e-14,
):
    """Run the full state-cache, geometry, link, and assembly pipeline."""
    state_dir, _ = RunExcitonStateCacheParallel(
        cfg=cfg,
        bse_dir=bse_dir,
        ni_wf_path=ni_wf_path,
        phase_data_path=phase_data_path,
        state_dir=state_dir,
        band_c=band_c,
        band_v=band_v,
        exciton_state_index=exciton_state_index,
        normalize=normalize,
        range_handle=range_handle,
        n_workers=n_workers,
        resume=resume,
    )
    return RunExcitonGeometryFromCacheParallel(
        cfg=cfg,
        state_dir=state_dir,
        geometry_dir=geometry_dir,
        output_path=output_path,
        dQ1=dQ1,
        dQ2=dQ2,
        n_workers=n_workers,
        resume=resume,
        overlap_tolerance=overlap_tolerance,
    )
#----------------------------------------------------------#
def GetBerryCurvature1band(u):
    N = u.shape[0] #assumes u.shape[N,N,:]
    link_x=np.zeros([N,N], dtype="complex")
    link_y=np.zeros([N,N], dtype="complex")
    BC=np.zeros([N,N], dtype="float")
    for i in range(N):
        for j in range(N):
            iL,jL,nxL,nyL = TE.PBCAddInd((i,j,0,0),(-1,0),N) #all the n's are unused because everything here is in the first BZ
            iD,jD,nxD,nyD = TE.PBCAddInd((i,j,0,0),(0,-1),N)
            num = np.vdot(u[iL, jL], u[i, j])
            link_x[i, j] = num / np.abs(num)
            num = np.vdot(u[iD, jD], u[i, j])
            link_y[i, j] = num / np.abs(num)
    for i in range(N):
        for j in range(N):
            iL,jL,nxL,nyL = TE.PBCAddInd((i,j,0,0),(-1,0),N)
            iD,jD,nxD,nyD = TE.PBCAddInd((i,j,0,0),(0,-1),N)
            val = np.imag(np.log(link_x[i,j]*link_y[iL,jL]/(link_x[iD,jD]*link_y[i,j])))
            BC[i,j] = val
    C=np.sum(BC)/(2*np.pi)
    return BC,C

def GetExcitonBC(Fn, ni_wf,v,c,phase_data,range_handle):
    """
    accepts Fn[Qx,Qy,px,py] (Qband is selected already)
    """
    NQ = Fn.shape[0]
    link_x=np.zeros([NQ,NQ], dtype="complex")
    link_y=np.zeros([NQ,NQ], dtype="complex")
    BC=np.zeros([NQ,NQ], dtype="float")
    for i in range(NQ):
        for j in range(NQ):
            KQ=(i,j,0,0)
            KQL = TE.PBCAddInd(KQ,(-1,0),NQ)
            KQD = TE.PBCAddInd(KQ,(0,-1),NQ)
            num = ExcitonVProd_PBasis((KQ[0],KQ[1]),(KQL[0],KQL[1]),Fn,ni_wf,v,c,phase_data,range_handle)
            link_x[i, j] = num / np.abs(num)
            num = ExcitonVProd_PBasis((KQ[0],KQ[1]),(KQD[0],KQD[1]),Fn,ni_wf,v,c,phase_data,range_handle)
            link_y[i, j] = num / np.abs(num)
    for i in range(NQ):
        for j in range(NQ):
            KQ=(i,j,0,0)
            KQL = TE.PBCAddInd(KQ,(-1,0),NQ)
            KQD = TE.PBCAddInd(KQ,(0,-1),NQ)
            val = np.imag(np.log(link_x[KQ[0],KQ[1]]*link_y[KQL[0],KQL[1]]/(link_x[KQD[0],KQD[1]]*link_y[KQ[0],KQ[1]])))
            BC[i,j] = val
    C=np.sum(BC)/(2*np.pi)
    return BC,C

def GetExcitonProbability(Fn,ni_wf,v,c,phase_data,range_handle):
    """
    accepts Fn[Qx,Qy,px,py] (Qband is selected already)
    """
    NQ = Fn.shape[0]
    P = np.zeros([NQ,NQ],dtype = 'f')
    for i in range(NQ):
        for j in range(NQ):
            P[i,j] = ExcitonVProd_PBasis((i,j),(i,j),Fn,ni_wf,v,c,phase_data,range_handle)
    return P

def ExcitonVProd_PBasis(Q,Qp,Fn,ni_wf,v,c,phase_data,range_handle,seam_across_k1=False):
    """
    accepts Q and Qp integer touples, Fn[Qx,Qy,px,py] ni_wf[kx,ky,orb,band]
    v,c band indices, phase_data[g1[ky,band] sewing phase, U_g_arr [orb,orb] orbital sewing phase],
    seam_across_k1 for single particle phase seam direction
    """
    nk = ni_wf.shape[0]
    nQ = Fn.shape[0]
    val =0
    if range_handle == "forward":
        qxi = Q[0]
        qyi = Q[1]
        qxip = Qp[0]
        qyip = Qp[1]          
    elif range_handle == "center":
        qxi = Q[0] - nQ//2
        qyi = Q[1] - nQ//2
        qxip = Qp[0] - nQ//2
        qyip = Qp[1] - nQ//2
    
    else:
        raise ValueError(
            f"range_handle must be 'forward' or 'center', got {range_handle}"
        )
    for px in range(nk):
        for py in range(nk):
            Kp = (px,py,0,0)
            Kc = TE.PBCAddInd(Kp,(qxi,qyi),nk)
            Kv = TE.PBCAddInd(Kp,(-qxi,-qyi),nk)
            Kcp = TE.PBCAddInd(Kp,(qxip,qyip),nk)
            Kvp = TE.PBCAddInd(Kp,(-qxip,-qyip),nk)
            FQ = Fn[Q[0],Q[1],Kp[0],Kp[1]]
            FQp = Fn[Qp[0],Qp[1],Kp[0],Kp[1]]
            uc = TE.u_ext(ni_wf,phase_data,Kc,c,seam_across_k1)
            uv = TE.u_ext(ni_wf,phase_data,Kv,v,seam_across_k1)
            ucp = TE.u_ext(ni_wf,phase_data,Kcp,c,seam_across_k1)
            uvp = TE.u_ext(ni_wf,phase_data,Kvp,v,seam_across_k1)
            val+= np.vdot(FQp,FQ)*np.vdot(ucp,uc)*np.vdot(uv,uvp)
    return val

def GetElectronHoleOverlaps(ni_wf, phase_data, dQ=(0,0), NQ=32, Np=32, c_band = 1, v_band = 0):
    """
    construct S_p(Q,Q') = <p;Q|p;Q'> = <u_c(p+Q/2)|u_c(p+Q'/2)><u_v(p-Q'/2)|u_v(p+Q/2)>
    here Q' = Q+dQ , dQ is expressed in indices in the Q1,Q2 reciprocal lattice directions.
    returns an array [Q1,Q2,p] where for each (Q1,Q2) coordinate a p matrix is expressed.
    """
    S_p = np.zeros((NQ,NQ,Np,Np), dtype = complex) #flat p coordinates
    c_overlaps = np.zeros_like(S_p)
    v_overlaps = np.zeros_like(S_p)
    for Q1 in range(NQ):
        Q1p = Q1+dQ[0]
        for Q2 in range(NQ):
            Q2p = Q2+dQ[1]
            for p1 in range(Np):
                for p2 in range(Np):
                    if phase_data[0] is not None:
                        Kp = (p1,p2,0,0)
                        Kv = TE.PBCAddInd(Kp,(-Q1,-Q2),Np)
                        Kc = TE.PBCAddInd(Kp,(Q1,Q2),Np)
                        Kvp = TE.PBCAddInd(Kp,(-Q1p,-Q2p),Np)
                        Kcp = TE.PBCAddInd(Kp,(Q1p,Q2p),Np)
                        uc = TE.u_ext(ni_wf,phase_data,Kc,c_band,U_arr)
                        uv = TE.u_ext(ni_wf,phase_data,Kv,v_band,U_arr)
                        ucp = TE.u_ext(ni_wf,phase_data,Kcp,c_band,U_arr)
                        uvp = TE.u_ext(ni_wf,phase_data,Kvp,v_band,U_arr)
                    else:
                        kv = ((p1-Q1)%NQ,(p2-Q2)%NQ)
                        kc = ((p1+Q1)%NQ,(p2+Q2)%NQ)
                        kvp = ((p1-Q1p)%NQ,(p2-Q2p)%NQ)
                        kcp = ((p1+Q1p)%NQ,(p2+Q2p)%NQ)
                        uv = ni_wf[kv[0],kv[1],:,v_band]
                        uc = ni_wf[kc[0],kc[1],:,c_band]
                        uvp = ni_wf[kvp[0],kvp[1],:,v_band]
                        ucp = ni_wf[kcp[0],kcp[1],:,c_band]
                    #c_overlaps[Q1,Q2,p1,p2] = np.vdot(uc,ucp)
                    #v_overlaps[Q1,Q2,p1,p2] = np.vdot(uv,uvp)
                    S_p[Q1,Q2,p1,p2] = np.vdot(uc,ucp)*np.vdot(uvp,uv)
    return S_p# , c_overlaps, v_overlaps

def SaveData(data,filename,folder):
    if not filename.endswith('.npy'):
        filename += '.npy'
    full_path = folder / filename
    np.save(full_path, data)

def MainProjectorGeometry(
    cfg_dir,
    band,
    storage_path,
    down_folder=None,
    n_workers=None,
):
    data_path = Path(storage_path) / Path(cfg_dir)
    job_path = data_path.parent
    bse_dir = job_path / Path("perQ")
    phase_data_path = job_path / Path("inputs/phase_data.npz")
    ni_wf_path = job_path / Path("inputs/ni_wf.npy")
    cfg = TE.GetConfig(data_path / "config.yml")

    NQ = int(cfg["NQ"])
    c_band = int(cfg["c_band"])
    v_band = int(cfg["v_band"])
    b1 = np.asarray(cfg["rec_vec"][0], dtype=float)
    b2 = np.asarray(cfg["rec_vec"][1], dtype=float)
    range_handle = cfg["Grid"]["Range"]

    # These are true for the extended handleing
    dQ1 = 2*np.linalg.norm(b1) / NQ
    dQ2 = 2*np.linalg.norm(b2) / NQ

    state_dir = job_path / "ExcitonStates_U[{}]".format(band)
    geometry_dir = job_path / "ExcitonGeometry_U[{}]_per_Q".format(band)
    output_path = data_path  / "ExcitonGeometry_U[{}].npz".format(band)

    n_workers = TE.ResolveWorkerCount(cfg)
    result_path = RunExcitonGeometryPipeline(
        cfg=cfg,
        bse_dir=Path(bse_dir),
        ni_wf_path=Path(ni_wf_path),
        phase_data_path=Path(phase_data_path),
        state_dir=state_dir,
        geometry_dir=geometry_dir,
        output_path=output_path,
        band_c=c_band,
        band_v=v_band,
        exciton_state_index=band,
        dQ1=dQ1,
        dQ2=dQ2,
        normalize=True,
        range_handle=range_handle,
        n_workers=n_workers,
        resume=True,
    )

    with np.load(result_path, allow_pickle=False) as result:
        print("Projector Chern number:", float(result["C_projector"]))
        print("Link Chern number:", float(result["C_links"]))

    if down_folder:
        down_path = Path(storage_path) / "Download" / down_folder
        DataManager.CopyDirectory(data_path, down_path)

    return result_path

def MainFTrans(cfg_dir,band,storage_path):
    #WRONG
    config_folder = Path(f"{cfg_dir}")
    config_name = Path("config.yml")
    f_file = storage_path/config_folder/Path(f"F[{band}].npy")
    config_file = storage_path/config_folder/config_name
    cfg = TE.GetConfig(config_file)
    NQ = cfg["NQ"]
    Np = cfg["Nk"]
    F0 = np.load(f_file, mmap_mode='r')
    F0_flat = F0.reshape(NQ, NQ, Np*Np)
    _, ni_wf = TE.SolveBzNoninteracting(cfg)
    phase_data = TE.SeamHandle(ni_wf,cfg)
    dQ = (1,0)
    S_p = GetElectronHoleOverlaps(ni_wf,g1,dQ=dQ)
    print("finished overlaps")
    F_transport = np.zeros_like(F0_flat, dtype=complex)
    
    for Q1 in range(NQ):
        for Q2 in range(NQ):
            Q1p = (Q1 + dQ[0]) % NQ
            Q2p = (Q2 + dQ[1]) % NQ
            rhs = S_ppt[Q1, Q2] @ F0_flat[Q1p, Q2p]
            F_transport[Q1, Q2] = np.linalg.solve(
                G_ppt[Q1, Q2],
                rhs
            )
    F_trans = F_transport.reshape(NQ,NQ,Np,Np)
    SaveData(F_trans, f"F_trans",storage_path/config_folder)
    return
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    #ap.add_argument("--dir_parent", required=True, help="Input directory")
    ap.add_argument("--dir", required=True, help="Input directory")
    ap.add_argument("--band", required=True,type=int, help="Input desired exciton band")
    ap.add_argument("--down_folder",default = None, required = False, help = "Directory name in Download folder")    
    args = ap.parse_args()
    storage_path = Path("/tobiasholder-storage/ofirbarogel")
    cfg_dir = args.dir
    band = args.band
    down_folder = args.down_folder
    MainProjectorGeometry(cfg_dir=cfg_dir,band=band,storage_path=storage_path,down_folder = down_folder)
    #MainFTrans(cfg_dir,band,storage_path)
        