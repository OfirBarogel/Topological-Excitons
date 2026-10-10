import numpy as np
import time
import os
from pathlib import Path
import argparse
from datetime import datetime
from typing import List
from zipfile import BadZipFile
import yaml
import ExcitonWF
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import DataManager
from scipy.linalg import eigh
#ver 0.1.8
def GetConfig(filename):
    with open(filename, 'r') as stream:
        try:
            cfg = yaml.load(stream)
        except TypeError:
            cfg = yaml.load(stream, Loader=yaml.UnsafeLoader)
    hc = cfg["hamiltonian"]
    hkind = hc["kind"]
    if hkind in{"haldane","mos2"}:
        ltype =  "hexagonal"
    elif hkind in {"qwz", "qwz analytic"}:
        ltype = "square"
    else:
        raise ValueError(f"Lattice type not configured: \"{hkind}\"")
    hc["lattice"] = ltype
    terms: List[str] = [t for t in hc.get("terms", [])]
    if hkind in {'qwz', "haldane", "qwz analytic"}:
        cfg["Bands"] = 2
    elif hkind == 'mos2':
        cfg["Bands"] = 6 if any(t in {"zeeman", "soc"} for t in terms) else 3
    HandleGridConfig(cfg)
    return cfg

def HandleGridConfig(cfg):
    NQ = cfg["NQ"]
    grid_cfg = cfg.get("Grid")
    grid_handle = grid_cfg.get("Handle","extend")
    range_handle = grid_cfg.get("Range","center")
    ltype = cfg["hamiltonian"]["lattice"]
    bands = cfg["Bands"]
    if grid_handle == "extend":
        Nk = NQ
        Q_range_mult = 2
    elif grid_handle == "sew":
        Nk = 2*NQ
        Q_range_mult = 1
    if range_handle == "forward":
        k_frac = np.linspace(0,1,Nk,endpoint=False)
        Q_frac = np.linspace(0,1*Q_range_mult,NQ,endpoint=False)
    elif range_handle == "center":
        k_frac = np.linspace(-0.5,0.5,Nk,endpoint=False)
        Q_frac = np.linspace(-0.5*Q_range_mult,0.5*Q_range_mult,NQ,endpoint=False)
    else:
        raise Exception("wrong range handle")
    if ltype == "square":
        b1 = (2*np.pi,0)
        b2 = (0,2*np.pi)
        if bands == 2:
            tau = np.array([[0,0],[0,0]])
        else:
            raise Exception("multiband square lattice orbital positions not configured")
    elif ltype == "hexagonal":        
        b1 = (2*np.pi/3,2*np.pi/np.sqrt(3))
        b2 = (-2*np.pi/3,2*np.pi/np.sqrt(3))
        if bands ==2:
            tau = np.array([[0,0],[-1,0]])
        else:
            raise Exception("multiband hexagonal lattice orbital positions not configured")
    b = [np.array(b1), np.array(b2)]
    cfg["tau"] = tau
    cfg["Nk"] = Nk
    cfg["rec_vec"] = b
    cfg["k_frac"] = k_frac
    cfg["Q_frac"] = Q_frac
    return

#for the py file
def SaveConfig(config_file,cfg):
    keys_to_save = ["version", "hamiltonian", "NQ", "Grid", "U", "epsilon","interaction","c_band","v_band","eps1","eps2","n_Fs","RunType"]
    config = {key: cfg.get(key) for key in keys_to_save if key in cfg}
    with open(config_file, 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

def GetAnalyticQWZ(kx,ky,M):
    dx = np.sin(kx)
    dy = np.sin(ky)
    dz = M + np.cos(kx) + np.cos(ky)
    d  = np.sqrt(dx*dx + dy*dy + dz*dz)

    uvA = np.array([dz - d, dx + 1j*dy], dtype=complex)        # bad near d≈dz (north pole)
    uvB = np.array([dx - 1j*dy, -(dz + d)], dtype=complex)     # bad near d≈-dz (south pole)
    
    if abs(d + dz) > abs(dx - 1j*dy):
        uc = np.array([d + dz, dx + 1j*dy], dtype=complex)
    else:
        uc = np.array([dx - 1j*dy, d - dz], dtype=complex)

    # --- Valence band (E = -d) ---
    if abs(d - dz) > abs(dx - 1j*dy):
        uv = np.array([dz - d, dx + 1j*dy], dtype=complex)
    else:
        uv = np.array([dx - 1j*dy, -(d + dz)], dtype=complex)

    # Normalize
    uc /= np.linalg.norm(uc)
    uv /= np.linalg.norm(uv)

    evals = np.array([-d, d])
    evecs = np.column_stack((uv, uc))
    
    return evals, evecs

def QWZHamiltonian(kx,ky,M, t = -1, tt = 0):
    pauli_mat =[np.array([[0,1],[1,0]]), np.array([[0,-1j],[1j,0]]), np.array([[1,0],[0,-1]])]
    I = np.array([[1,0],[0,1]])
    return -tt*(np.cos(kx)+np.cos(ky))*I+np.sin(kx)*pauli_mat[0] + np.sin(ky)*pauli_mat[1] + (M-t*(np.cos(kx)+np.cos(ky)))*pauli_mat[2]

def HaldaneH(kx, ky, t1=1.0, t2=0.1, phi=0, M=0, a=1.0):
    d = np.array([
        [-a, 0.0],
        [ a/2.0, -np.sqrt(3)*a/2.0],
        [ a/2.0,  np.sqrt(3)*a/2.0],
    ])
    b = np.vstack([d[1] - d[2] , d[2] - d[0],d[0] - d[1]])
    k = np.array([kx, ky])
    f = np.exp(1j * (k @ d.T)).sum()
    cb = np.cos(k @ b.T).sum()
    sb = np.sin(k @ b.T).sum()
    d0 = 2.0 * t2 * np.cos(phi) * cb
    dz = M - 2.0 * t2 * np.sin(phi) * sb
    H = np.array([[d0 + dz, -t1 * f],
                  [-t1 * np.conj(f), d0 - dz]], dtype=complex)
    return H
# 10.1103/PhysRevB.88.085433
def TNNMoS_2Hamiltonian(kx,ky,terms,params):
    has_soc    = "soc" in terms
    has_zeeman = "zeeman" in terms
    alpha =0.5*kx
    beta = (np.sqrt(3)/2)*ky
    epsilon1 =0.683
    epsilon2 =1.707
    t0 = -0.146
    t1 = -0.114
    t2 = 0.506
    t11 = 0.085
    t12 = 0.162
    t22 = 0.073
    r0 = 0.060
    r1 = -0.236
    r2 = 0.067
    r11 = 0.016
    r12 = 0.087
    u0 = -0.038
    u1 = 0.046
    u2 = 0.001
    u11 = 0.266
    u12 = -0.176
    u22 = -0.150
    if "modified" in terms:
        delta = params.get("modified").get("delta",0.0)
    else:
        delta = 0.0
    h0 = (epsilon1 +
          2 * t0 * (2 * np.cos(alpha) * np.cos(beta) + np.cos(2 * alpha)) +
          2 * r0 * (2 * np.cos(3 * alpha) * np.cos(beta) + np.cos(2 * beta)) +
          2 * u0 * (2 * np.cos(2 * alpha) * np.cos(2 * beta) + np.cos(4 * alpha)))

    h1_real = (-2 * np.sqrt(3) * t2 * np.sin(alpha) * np.sin(beta) +
               2 * (r1 + r2) * np.sin(3 * alpha) * np.sin(beta) -
               2 * np.sqrt(3) * u2 * np.sin(2 * alpha) * np.sin(2 * beta))

    h1_imag = (2 * t1 * np.sin(alpha) * (2 * np.cos(alpha) + np.cos(beta)) +
               2 * (r1 - r2) * np.sin(3 * alpha) * np.cos(beta) +
               2 * u1 * np.sin(2 * alpha) * (2 * np.cos(2 * alpha) + np.cos(2 * beta)))

    h1 = h1_real + 1j * h1_imag

    h2_real = (2 * t2 * (np.cos(2 * alpha) - np.cos(alpha) * np.cos(beta)) -
               (2 / np.sqrt(3)) * (r1 + r2) * (np.cos(3 * alpha) * np.cos(beta) - np.cos(2 * beta)) +
               2 * u2 * (np.cos(4 * alpha) - np.cos(2 * alpha) * np.cos(2 * beta)))

    h2_imag = (2 * np.sqrt(3) * t1 * np.cos(alpha) * np.sin(beta) +
               (2 / np.sqrt(3)) * np.sin(beta) * (r1 - r2) * (np.cos(3 * alpha) + 2 * np.cos(beta)) +
               2 * np.sqrt(3) * u1 * np.cos(2 * alpha) * np.sin(2 * beta))

    h2 = h2_real + 1j * h2_imag

    h11 = (epsilon2 +
           (t11 + 3 * t22) * np.cos(alpha) * np.cos(beta) +
           2 * t11 * np.cos(2 * alpha) +
           4 * r11 * np.cos(3 * alpha) * np.cos(beta) +
           2 * (r11 + np.sqrt(3) * r12) * np.cos(2 * beta) +
           (u11 + 3 * u22) * np.cos(2 * alpha) * np.cos(2 * beta) +
           2 * u11 * np.cos(4 * alpha)+delta)

    h12_real = (np.sqrt(3) * (t22 - t11) * np.sin(alpha) * np.sin(beta) +
                4 * r12 * np.sin(3 * alpha) * np.sin(beta) +
                np.sqrt(3) * (u22 - u11) * np.sin(2 * alpha) * np.sin(2 * beta))

    h12_imag = (4 * t12 * np.sin(alpha) * (np.cos(alpha) - np.cos(beta)) +
                4 * u12 * np.sin(2 * alpha) * (np.cos(2 * alpha) - np.cos(2 * beta)))

    h12 = h12_real + 1j * h12_imag

    h22 = (epsilon2 + 
           (3 * t11 + t22) * np.cos(alpha) * np.cos(beta) +
           2 * t22 * np.cos(2 * alpha) +
           2 * r11 * (2 * np.cos(3 * alpha) * np.cos(beta) + np.cos(2 * beta)) +
           (2 / np.sqrt(3)) * r12 * (4 * np.cos(3 * alpha) * np.cos(beta) - np.cos(2 * beta)) +
           (3 * u11 + u22) * np.cos(2 * alpha) * np.cos(2 * beta) +
           2 * u22 * np.cos(4 * alpha)-delta)

    # Construct the Hamiltonian matrix
    H_0 = np.array([
        [h0, h1, h2],
        [np.conj(h1), h11, h12],
        [np.conj(h2), np.conj(h12), h22]
    ])
    if not has_soc and not has_zeeman:
        return H_0
    sx = np.array([[0., 1.],
                   [1., 0. ]], dtype=complex)
    sy = np.array([[0., -1j],
                   [1j,  0. ]], dtype=complex)
    sz = np.array([[1., 0. ],
                   [0., -1.]], dtype=complex)

    # Base in spinful space: diag(H0, H0)  (6x6)
    H_spinful = np.block([
        [H_0,               np.zeros_like(H_0)],
        [np.zeros_like(H_0), H_0             ]
    ])
    if has_zeeman:
        B = np.array(params.get("zeeman", {}).get("B", [0.0, 0.0, 0.0]), dtype=float)
        if B.shape != (3,):
            raise ValueError("Zeeman.B must be a length-3 array/list [Bx, By, Bz].")
        H_spin = B[0]*sx + B[1]*sy + B[2]*sz           # 2x2
        H_zeeman = np.kron(np.eye(3, dtype=complex), H_spin)  # 6x6
        H_spinful = H_spinful + H_zeeman
    if has_soc:
        # lambda_soc from config if present; fallback to 0.073
        lambda_soc = params.get("soc", {}).get("strength", 0.073)
        Lz = np.array([[0,  0,   0 ],
                       [0,  0,  2j ],
                       [0, -2j,  0 ]], dtype=complex)

        SOC_block = (lambda_soc / 2.0) * Lz  # 3x3
        # Expand to spin basis: +SOC on ↑, -SOC on ↓
        SOC_full = np.block([
            [ SOC_block,               np.zeros_like(SOC_block)],
            [ np.zeros_like(SOC_block), -SOC_block            ]
        ])
        H_spinful = H_spinful + SOC_full
    return H_spinful

def PBCAddInd(k_wrap,delta,N):
    """
    k_wrap: (i,j,nx,ny) representing the coord in the first BZ and how many wraps on each
    delta: just id,jd
    works in all lattice types since using the parallelogram BZ (acts like square when using i*b1+j*b2)
    """
    i,j,nx,ny = k_wrap
    di, dj = delta
    sx = i + di
    sy = j + dj
    dnx = sx // N
    dny = sy // N
    ix = sx % N
    jy = sy % N
    return ix, jy, nx+dnx, ny+dny

def UnitPhase(z, eps=1e-14):
    a = np.abs(z)
    return z / (a + eps)

def ParallelTransportCylindrical(
    u,
    enforce_ky_periodic=True,
    swap_k_coordinates=False,
):
    """
    u: shape (Nx, Ny, Norb, Nband), normalized.

    If swap_k_coordinates=False:
      - parallel transport coordinate: k1 / x
      - periodic coordinate: k2 / y

    If swap_k_coordinates=True:
      - parallel transport coordinate: k2 / y
      - periodic coordinate: k1 / x

    note: for hexagonal the coordinates of u are u[b1,b2,orb,band]
    """

    if swap_k_coordinates:
        # Work in swapped coordinates internally:
        # u_work[a,b] = u[b,a]
        u_work = np.swapaxes(u, 0, 1)
    else:
        u_work = u

    u_g_work = np.zeros_like(u_work, dtype=np.complex128)
    Nx, Ny, Norb, Nband = u_work.shape

    for band in range(Nband):
        # 1) Seed at (0,0)
        u_g_work[0, 0, :, band] = u_work[0, 0, :, band]

        # 2) Parallel transport along periodic coordinate on first column
        for j in range(Ny - 1):
            z = np.vdot(
                u_g_work[0, j, :, band],
                u_work[0, j + 1, :, band],
            )
            u_g_work[0, j + 1, :, band] = (
                np.conj(UnitPhase(z)) * u_work[0, j + 1, :, band]
            )

        # 3) Enforce trivial sewing on first column
        if enforce_ky_periodic:
            z_end = np.vdot(
                u_g_work[0, Ny - 1, :, band],
                u_g_work[0, 0, :, band],
            )
            theta = np.angle(z_end)

            for j in range(Ny):
                u_g_work[0, j, :, band] *= np.exp(-1j * j * theta / Ny)

        # 4) Parallel transport along transport coordinate
        for j in range(Ny):
            for i in range(Nx - 1):
                z = np.vdot(
                    u_g_work[i, j, :, band],
                    u_work[i + 1, j, :, band],
                )
                u_g_work[i + 1, j, :, band] = (
                    np.conj(UnitPhase(z)) * u_work[i + 1, j, :, band]
                )

    if swap_k_coordinates:
        return np.swapaxes(u_g_work, 0, 1)

    return u_g_work

def SeamPhaseG(u_g, seam_across_kx=False):
    """
    u_g: (Nx, Ny, Norb, Nband)

    If seam_across_kx=False:
      returns g1[j, band], shape (Ny, Nband)
      seam compares k1 end to k1 start at fixed k2:
          <u_g[Nx-1, j] | u_g[0, j]>

    If seam_across_kx=True:
      returns gy[i, band], shape (Nx, Nband)
      seam compares k2 end to k2 start at fixed k1:
          <u_g[i, Ny-1] | u_g[i, 0]>
    """

    Nx, Ny, _, Nband = u_g.shape

    if not seam_across_kx:
        g1 = np.empty((Ny, Nband), dtype=complex)

        for band in range(Nband):
            for j in range(Ny):
                z = np.vdot(
                    u_g[Nx - 1, j, :, band],
                    u_g[0, j, :, band],
                )
                g1[j, band] = z / (np.abs(z) + 1e-14)

        return g1

    else:
        g2 = np.empty((Nx, Nband), dtype=complex)

        for band in range(Nband):
            for i in range(Nx):
                z = np.vdot(
                    u_g[i, Ny - 1, :, band],
                    u_g[i, 0, :, band],
                )
                g2[i, band] = z / (np.abs(z) + 1e-14)

        return g2

def PrecomputeUg(rec_vec, tau, max_n1=4, max_n2=4):
    """
    Precompute the combined orbital phase for every (n1, n2).
    Ug[n1 + max_n1, n2 + max_n2, orbital]
    """
    n_orb = tau.shape[0]

    Ug = np.empty(
        (2*max_n1 + 1, 2*max_n2 + 1, n_orb),
        dtype=complex
    )

    for n1 in range(-max_n1, max_n1 + 1):
        for n2 in range(-max_n2, max_n2 + 1):
            G = n1 * rec_vec[0] + n2 * rec_vec[1]
            Ug[n1 + max_n1, n2 + max_n2, :] = np.exp(-1j * tau @ G)

    return Ug

def u_ext(u, phase_data, K, band, seam_across_kx=False):
    i, j, nx, ny = K

    gauge_seam = phase_data["g1"]
    Ug = phase_data["Ug"]

    max_n1 = (Ug.shape[0] - 1) // 2 #should be 4 in code, physically it should be 2 but better safe than sorry
    max_n2 = (Ug.shape[1] - 1) // 2

    if seam_across_kx:
        g_phase = gauge_seam[i, band] ** ny
    else:
        g_phase = gauge_seam[j, band] ** nx

    orbital_phase = Ug[nx + max_n1, ny + max_n2, :]

    return orbital_phase * g_phase * u[i, j, :, band]

def FirstRealGauge(v):
    first = 0
    if np.isclose(v[0], 0):
        first = 1

    phase = np.angle(v[first])
    v_fixed = v * np.exp(-1j * phase)

    return v_fixed

def GaugeFixFPerQ(FQ):
    """
    accepts F[P,n_keep] -> gauge fixes for each n level -> returns
    """
    Np_flat, n_keep  = FQ.shape
    FQ_gauged = np.zeros_like(FQ)
    for n in range(n_keep):
        Fn_gauged = FirstRealGauge(FQ[:,n])
        FQ_gauged[:,n] = Fn_gauged
    return FQ_gauged

def SolveH(kx,ky,hcfg):
    kind = hcfg.get("kind",None)
    params = hcfg.get("params")
    if kind == "mos2":
        terms = set(map(str.lower, hcfg.get("terms", [])))
        H = TNNMoS_2Hamiltonian(kx,ky,terms,params)
    elif kind == "qwz":
        M = params.get("M")
        t = params.get("t",-1)
        tt = params.get("tt",0)
        H = QWZHamiltonian(kx,ky,M,t,tt)
    elif kind =="haldane":
        M_=params.get("M")
        phi_ = params.get("phi")
        H = HaldaneH(kx,ky,phi=phi_,M=M_)
    elif kind =="qwz analytic":
        M = params.get("M")
        return GetAnalyticQWZ(kx,ky,M)
    else:
        raise ValueError("Could not parse acceptable Hamiltonian")
    #if we are here we are solving numerically
    val,evec = eigh(H)
    sorted_indices = np.argsort(val)
    sorted_val = val[sorted_indices]
    sorted_evec = evec[:, sorted_indices]
    return sorted_val,sorted_evec

def SolveBzNoninteracting(cfg):
    hcfg = cfg.get("hamiltonian")
    ltype = hcfg["lattice"]
    b = cfg["rec_vec"]
    k_frac = cfg["k_frac"]
    n_points = cfg["Nk"]
    n_bands = cfg["Bands"]
    dk_frac = 1/n_points
    eps1_factor = dk_frac*cfg.get("eps1",0)
    eps2_factor = dk_frac*cfg.get("eps2",0)
    (epsx,epsy) = b[0]*eps1_factor+b[1]*eps2_factor
    wf_BZ = np.zeros((n_points,n_points,n_bands,n_bands), dtype=complex) #k1,k2,orbital weight, band
    eval_BZ = np.zeros((n_points,n_points,n_bands))
    for i in range(n_points):
        for j in range(n_points):
            (kx,ky) = k_frac[i]*b[0] + k_frac[j]*b[1]
            (val,vec)=SolveH(kx+epsx,ky+epsy,hcfg)
            eval_BZ[i,j] = val
            wf_BZ[i,j] = vec
    wf_BZ[:,:,:,:] = ParallelTransportCylindrical(wf_BZ[:,:,:,:])
    return eval_BZ, wf_BZ

def SeamHandle(ni_wf,cfg):
    b= cfg["rec_vec"]
    tau = cfg["tau"]
    g1 = SeamPhaseG(ni_wf)
    Ug = PrecomputeUg(b,tau)
    phase_data = {
        "g1": g1,
        "Ug": Ug
    }
    return phase_data

def GetH2Band(Q1,Q2,ni_wf,ni_evals,cfg,phase_data): 
    U = cfg["U"]
    epsilon = cfg["epsilon"]
    [b1,b2] = cfg["rec_vec"]
    k_frac = cfg["k_frac"]
    Nk = cfg["Nk"]
    range_handle = cfg["Grid"]["Range"]
    interaction = cfg["interaction"]
    params = interaction["params"]
    c_band = cfg["c_band"]
    v_band = cfg["v_band"]
    #weight for integral from Jacobian and dk1 dk2
    Jacobian = np.abs(b1[0]*b2[1] - b1[1]*b2[0])
    wk = Jacobian/pow(Nk,2)
    if range_handle == "forward":
        q1i = Q1
        q2i = Q2
    elif range_handle == "center":
        NQ = cfg["NQ"]
        q1i = Q1 - NQ//2
        q2i = Q2 - NQ//2        
    H = np.zeros((Nk,Nk,Nk,Nk), dtype = complex) #p1,p2,p1',p2'
    for p1 in range(Nk):
        for p2 in range(Nk):
            Kp = (p1,p2,0,0)
            Kv = PBCAddInd(Kp,(-q1i,-q2i),Nk)
            Kc = PBCAddInd(Kp,(q1i,q2i),Nk)
            for p1p in range(Nk):
                for p2p in range(Nk):
                    Kpp = (p1p,p2p,0,0)
                    Kvp = PBCAddInd(Kpp,(-q1i,-q2i),Nk)
                    Kcp = PBCAddInd(Kpp,(q1i,q2i),Nk)
                    if p1==p1p and p2==p2p:
                        e = ni_evals[Kc[0]][Kc[1]][c_band]- ni_evals[Kv[0]][Kv[1]][v_band]
                    else:
                        e = 0
                    if interaction.get("kind") == "coulomb":
                        Q_frac = cfg["Q_frac"]
                        V_d_weight = params.get('vd',1)
                        V_x_weight = params.get('vx',1)
                        k_vec = np.array(k_frac[Kv[0]]*b1 + k_frac[Kv[1]]*b2)
                        k_vecp = np.array(k_frac[Kvp[0]]*b1 + k_frac[Kvp[1]]*b2)
                        deltak_vec = k_vec-k_vecp
                        deltak_norm  = np.linalg.norm(deltak_vec)
                        V_d = V_d_weight/(deltak_norm+epsilon)
                        if p1==p1p and p2==p2p:
                            V_d = 0
                        Q_vec = np.array(Q_frac[Q1]*b1 + Q_frac[Q2]*b2)
                        Q_norm  = np.linalg.norm(Q_vec)
                        V_x = V_x_weight/(Q_norm+epsilon)
                    elif interaction.get("kind") == "constant":
                        V_d = params.get('vd')
                        V_x = params.get('vx')
                    elif interaction.get("kind") == "separable":
                        V = params.get('vx')
                        #<u_c(p+Q/2)|u_c(p-Q/2)><u_v(p'-Q/2)|u_v(p'+Q/2)>
                        # <u_c(Kc)|u_c(Kv)><u_v(Kv)|u_v(Kc)>
                        uc_c = u_ext(ni_wf,phase_data,Kc,c_band)
                        uc_v = u_ext(ni_wf,phase_data,Kv,c_band,U_arr)
                        uv_vp = u_ext(ni_wf,phase_data,Kvp,v_band,U_arr)
                        uv_cp = u_ext(ni_wf,phase_data,Kcp,v_band,U_arr)
                        I = U*V*np.vdot(uc_c,uc_v)*np.vdot(uv_vp,uv_cp)
                        H[p1,p2,p1p,p2p] = e - wk*I
                        continue
                    else:
                        raise ValueError("Could not parse acceptable interaction kind")
                    uc = u_ext(ni_wf,phase_data,Kc,c_band)
                    uv = u_ext(ni_wf,phase_data,Kv,v_band)
                    ucp = u_ext(ni_wf,phase_data,Kcp,c_band)
                    uvp = u_ext(ni_wf,phase_data,Kvp,v_band)
                    D = U*V_d*np.vdot(uc, ucp) * np.vdot(uvp, uv)
                    X = U*V_x*np.vdot(uc, uv) * np.vdot(uvp, ucp)
                    H[p1,p2,p1p,p2p] = e - wk*(D-X)
    return H

def _SavePerQ(path, i, j, e_all, f_keep, meta=None):
    """Save per-Q results atomically."""
    path = Path(path)
    tmp = path / f"Q_{i:03d}_{j:03d}.tmp.npz"
    out = path / f"Q_{i:03d}_{j:03d}.npz"

    np.savez_compressed(tmp, e_all=e_all, f_keep=f_keep, meta=meta or {})
    os.replace(tmp, out)
    return out

#---Parallel code---#
def SolveOneQTask(Qx, Qy, cfg, ni_wf_path, ni_evals_path, phase_data_path, outdir):    
    """
    Worker function executed in each process.
    Qx, Qy, cfg_path, ni_wf_path, ni_evals_path, phase_data, outdir
    """
    n_Fs = cfg["n_Fs"]
    # Large arrays: mmap_mode="r" avoids copying; workers open read-only
    ni_wf = np.load(ni_wf_path, mmap_mode="r")
    ni_evals = np.load(ni_evals_path, mmap_mode="r")
    phase_data = np.load(phase_data_path, mmap_mode="r")

    # The function GetH2Band must accept these mmap arrays
    try:
        H_Q = GetH2Band(Qx, Qy, ni_wf, ni_evals, cfg, phase_data)
    except Exception as e:
        # propagate error in a readable way for the driver
        raise RuntimeError(f"GetH2Band failed for Q=({Qx},{Qy}): {e}")

    s = H_Q.shape
    Nk = cfg["Nk"]
    Nk2 = s[0] * s[1]  # should equal Nk*Nk
    H_Q2D = H_Q.reshape(Nk2, Nk2)

    # dense diagonalization; can be swapped later for eigsh/LinearOperator
    e_Q, f_Q = eigh(H_Q2D)
    sorted_indices = np.argsort(e_Q)
    sorted_e_Q = np.real(e_Q[sorted_indices])

    # Keep only n_Fs eigenvectors if requested
    if n_Fs is None or n_Fs <= 0:
        n_keep = Nk2
    else:
        n_keep = min(n_Fs, Nk2)

    sorted_f_Q = f_Q[:, sorted_indices[:n_keep]]   # shape (Nk2, n_keep)
    # reshape to (Nk, Nk, Nk2) if you want original px/py layout
    # But for compact storage we store flattened vectors: (Nk2, n_keep)
    f_keep = sorted_f_Q  # (Nk2, n_keep)
    #gauge fix the F's
    f_keep = GaugeFixFPerQ(f_keep)

    # Meta
    meta = {
        "Qx": int(Qx), "Qy": int(Qy),
        "Nk": int(Nk),
        "Nk2": int(Nk2),
        "NQ": int(cfg["NQ"]),
        "n_keep": int(n_keep)
    }

    # Save per-Q to outdir
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out = _SavePerQ(outdir, Qx, Qy, sorted_e_Q, f_keep, meta=meta)
    # Return path for logging
    return str(out)

def ResolveWorkerCount(cfg, n_workers=None):
    if n_workers is not None:
        return max(1, int(n_workers))

    if cfg.get("RunType", "Local") == "Local":
        return max(1, multiprocessing.cpu_count() - 1)

    if cfg.get("RunType") == "Cluster":
        return max(
            1,
            int(os.environ.get(
                "SLURM_CPUS_PER_TASK",
                multiprocessing.cpu_count(),
            )),
        )

    raise ValueError(f"Unknown RunType: {cfg.get('RunType')}")
def RunAllQParallel(cfg, ni_evals_path, ni_wf_path, phase_data_path, outdir,n_workers = None, resume=True):
    """
    Parallel driver. Saves per-Q `.npz` files in outdir.
    cfg: config dict (small enough to pickle/pass)
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    NQ = int(cfg["NQ"])

    # build list of Q points to run
    tasks = []
    for i in range(NQ):
        for j in range(NQ):
            perQ_file = outdir / f"Q_{i:03d}_{j:03d}.npz"
            if resume and perQ_file.exists():
                continue
            tasks.append((i, j))

    if not tasks:
        print("All Q tasks already exist; nothing to run.")
        return

    print(f"Running {len(tasks)} Q-points with {n_workers} workers...")

    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = [
            ex.submit(
                SolveOneQTask,
                i, j, cfg,
                str(ni_wf_path), str(ni_evals_path), str(phase_data_path), 
                str(outdir),
            )
            for (i, j) in tasks
        ]

        for fut in as_completed(futures):
            out = fut.result()   # will raise if worker raised
            print("done:", out)
#---Parallel code---#
def SaveData(data,filename,folder):
    if not filename.endswith('.npy'):
        filename += '.npy'
    full_path = folder / filename
    np.save(full_path, data)

def slurm_tag():
    env = os.environ
    # Prefer array IDs (job + task), else plain job id
    if env.get("SLURM_ARRAY_JOB_ID") and env.get("SLURM_ARRAY_TASK_ID"):
        base = f"{env['SLURM_ARRAY_JOB_ID']}.{env['SLURM_ARRAY_TASK_ID']}"
    elif env.get("SLURM_JOB_ID"):
        base = env["SLURM_JOB_ID"]
    else:
        return None

    # Optionally append step/proc for MPI steps
    step = env.get("SLURM_STEP_ID") or env.get("SLURM_PROCID")
    return f"slurm{base}" + (f"-s{step}" if step else "")

def HandlePaths(runtype,cfg_name,down_folder):
    if runtype == "Local":
        tag = ""
        storage_path = Path("Local_Tests")
    elif runtype == "Cluster":
        tag = slurm_tag()
        storage_path = Path("/tobiasholder-storage/ofirbarogel")
    data_folder = Path(f"Data/{cfg_name}-{tag}")
    data_path = storage_path / data_folder
    sub_data_path = data_path / Path(f"{cfg_name}")
    if down_folder:
        down_path = storage_path / Path("Download") / Path(down_folder)
    else:
        down_path = None
    return data_path, sub_data_path,down_path

def LoadOrCreateInputs(cfg, inputs_dir):
    """Load a complete input cache, or rebuild the whole cache if needed."""
    inputs_dir = Path(inputs_dir)
    inputs_dir.mkdir(parents=True, exist_ok=True)

    ni_wf_path = inputs_dir / "ni_wf.npy"
    ni_evals_path = inputs_dir / "ni_evals.npy"
    phase_data_path = inputs_dir / "phase_data.npz"

    try:
        # Open both arrays here so an incomplete or damaged cache is detected
        # before any BSE work is submitted.
        ni_wf = np.load(ni_wf_path, mmap_mode="r")
        ni_evals_check = np.load(ni_evals_path, mmap_mode="r")

        with np.load(phase_data_path) as saved:
            phase_data = {
                "g1": saved["g1"],
                "Ug": saved["Ug"],
            }

        Nk = int(cfg["Nk"])
        expected_grid_shape = (Nk, Nk)
        if ni_wf.shape[:2] != expected_grid_shape:
            raise ValueError(
                "Cached ni_wf grid does not match the stored config"
            )
        if ni_evals_check.shape[:2] != expected_grid_shape:
            raise ValueError(
                "Cached ni_evals grid does not match the stored config"
            )

        del ni_evals_check
        print("Using cached single-particle inputs.")

    except (
        FileNotFoundError,
        OSError,
        ValueError,
        KeyError,
        EOFError,
        BadZipFile,
    ) as exc:
        print(
            "Input cache is missing, incomplete, or incompatible; "
            f"rebuilding it ({exc})."
        )

        # Always rebuild all three files together so that the workers never
        # receive a mixture of inputs from different setup attempts.
        ni_evals, ni_wf = SolveBzNoninteracting(cfg)
        phase_data = SeamHandle(ni_wf, cfg)

        np.save(ni_wf_path, ni_wf)
        np.save(ni_evals_path, ni_evals)
        np.savez(
            phase_data_path,
            g1=phase_data["g1"],
            Ug=phase_data["Ug"],
        )

    return (
        ni_wf,
        phase_data,
        ni_evals_path,
        ni_wf_path,
        phase_data_path,
    )

def RunPipeline(
    cfg,
    data_path,
    sub_data_path,
    down_path=None,
    calculate_geometry=False,
):
    """Run every currently incomplete stage of an exciton calculation."""
    start = datetime.now()

    data_path = Path(data_path)
    sub_data_path = Path(sub_data_path)
    down_path = Path(down_path) if down_path is not None else None

    data_path.mkdir(parents=True, exist_ok=True)
    sub_data_path.mkdir(parents=True, exist_ok=True)

    runtime_path = data_path / "Runtime.txt"
    perq_dir = data_path / "perQ"
    inputs_dir = data_path / "inputs"
    perq_dir.mkdir(parents=True, exist_ok=True)

    (
        ni_wf,
        phase_data,
        ni_evals_path,
        ni_wf_path,
        phase_data_path,
    ) = LoadOrCreateInputs(cfg, inputs_dir)

    task_time = datetime.now() - start
    with runtime_path.open("a") as f:
        f.write(f"Setup: {task_time}\n")

    # Existing Q files are skipped, so this handles both new and interrupted
    # BSE calculations.
    task_time = datetime.now()
    n_workers = ResolveWorkerCount(cfg)
    RunAllQParallel(
        cfg,
        ni_evals_path,
        ni_wf_path,
        phase_data_path,
        perq_dir,
        n_workers=n_workers,
        resume=True,
    )
    task_time = datetime.now() - task_time
    with runtime_path.open("a") as f:
        f.write(f"BSE: {task_time}\n")

    # Stitching and later outputs are safe to regenerate if a previous job
    # stopped after completing the BSE stage.
    NQ = cfg["NQ"]
    b1 = cfg["rec_vec"][0]
    b2 = cfg["rec_vec"][1]
    dQ1 = 2*np.linalg.norm(b1)/NQ
    dQ2 = 2*np.linalg.norm(b2)/NQ
    range_handle = cfg["Grid"]["Range"]
    n_Fs = cfg["n_Fs"]
    Nk = cfg["Nk"]
    Nk2 = Nk * Nk

    # 1) full eigenvalue stitching (cheapish)
    evals_all_path = sub_data_path / "Exciton_Energys.npy"
    DataManager.StitchEvals(perq_dir, evals_all_path, NQ)

    # 2) eigenvectors only up to n_Fs (only if n_Fs>0)
    evecs_keep_path = sub_data_path / "F_keep.npy"
    DataManager.StitchEvecs(perq_dir, evecs_keep_path, NQ, n_Fs) #Saves F[NQ,NQ,Nk^2,n_Fs]

    if n_Fs and n_Fs > 0:
        # Only needed for saving F[n] and running the older Vprod method.
        Fmem = np.load(evecs_keep_path, mmap_mode="r")

        v = int(cfg["v_band"])
        c = int(cfg["c_band"])
        for n in range(n_Fs):
            task_time = datetime.now()
            # ============================================================
            # 1. Optional: preserve F[n] and the Vprod method
            # ============================================================

            Fn_flat = Fmem[:, :, :, n]
            Fn = Fn_flat.reshape(NQ, NQ, Nk, Nk)

            SaveData(Fn, "F[{}]".format(n), sub_data_path)

            UnBCVprod, CnVprod = ExcitonWF.GetExcitonBC(
                Fn,
                ni_wf,
                v,
                c,
                phase_data,
                range_handle,
            )

            SaveData(
                UnBCVprod,
                "Vprod_U[{}]BC_C_{}".format(n, CnVprod),
                sub_data_path,
            )

            # Fn is no longer needed by the projector/link pipeline.
            del Fn
            del Fn_flat
            task_time = datetime.now() - task_time
            with runtime_path.open("a") as f:
                f.write(f"Vprod BC {n}: {task_time}\n")
            # ============================================================
            # 2. New parallel state-cache and geometry pipeline
            # ============================================================
            if calculate_geometry:
                task_time = datetime.now()
                state_dir = (
                    data_path
                    / "ExcitonStates_U[{}]".format(n)
                )

                geometry_per_q_dir = (
                    data_path
                    / "ExcitonGeometry_U[{}]_per_Q".format(n)
                )

                geometry_output_path = (
                    sub_data_path
                    / "ExcitonGeometry_U[{}].npz".format(n)
                )

                result_path = ExcitonWF.RunExcitonGeometryPipeline(
                    cfg=cfg,

                    # Original BSE per-Q files containing f_keep.
                    bse_dir=perq_dir,

                    # Same single-particle input files used by BSE workers.
                    ni_wf_path=ni_wf_path,
                    phase_data_path=phase_data_path,

                    # New per-band output locations.
                    state_dir=state_dir,
                    geometry_dir=geometry_per_q_dir,
                    output_path=geometry_output_path,

                    band_c=c,
                    band_v=v,

                    # Select f_keep[:, n].
                    exciton_state_index=n,

                    dQ1=dQ1,
                    dQ2=dQ2,

                    normalize=True,
                    range_handle=range_handle,

                    n_workers=n_workers,
                    resume=True,
                )
                task_time = datetime.now()-task_time
                with runtime_path.open("a") as f:
                    f.write(f"Geometry {n}: {task_time}\n")

    if down_path:
        DataManager.CopyDirectory(sub_data_path,down_path/sub_data_path.name)
    elapsed = datetime.now()-start
    with runtime_path.open("a") as f:
        f.write(f"Total: {elapsed}\n")

def Main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to config.yml")
    ap.add_argument("--down_folder",default = None, required = False, help = "Directory name in Download folder")
    args = ap.parse_args()
    down_folder = args.down_folder
    cfg = GetConfig(args.config)
    cfg_name = Path(args.config).stem
    calculate_geometry = cfg["Grid"]["SeamTest"] #SeamTest has been reporposed for calculate geometry bool
    # Handle directiries
    runtype = cfg["RunType"]
    data_path, sub_data_path, down_path = HandlePaths(runtype,cfg_name,down_folder)
    os.makedirs(data_path, exist_ok=True)
    os.makedirs(sub_data_path, exist_ok=True)
    config_file = sub_data_path / "config.yml"
    SaveConfig(config_file,cfg)
    RunPipeline(
        cfg=cfg,
        data_path=data_path,
        sub_data_path=sub_data_path,
        down_path=down_path,
        calculate_geometry=calculate_geometry,
    )

if __name__ == "__main__":
    Main()
