import numpy as np
import time
import os
from pathlib import Path
import argparse
from datetime import datetime
import yaml
import ExcitonWF
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import DataManager
#ver 0.1.5
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
    grid_handle = cfg.get("Grid","sew")
    range_handle = cfg.get("Range","center")
    ltype = cfg["hamiltonian"]["lattice"]
    if ltype == "hexagonal":
        raise ValueError("hexagonal grid handling not implemented")
    elif ltype == "square":
        if grid_handle == "extend":
            n_mult = 1
            Q_range_mult = 2
        elif grid_handle == "sew":
            n_mult = 2
            Q_range_mult = 1
        if range_handle == "foreward":
            kx_min = ky_min = 0
            kx_max = ky_max = 2*np.pi
        elif range_handle == "center":
            kx_min = ky_min = -np.pi
            kx_max = ky_max = np.pi
        Nk = n_mult*NQ
        #bc it is square im just multiplying kx range rather than both
        Qx_min = Qy_min = -Q_range_mult*kx_min
        Qx_max = Qy_max = Q_range_mult*kx_max
    cfg["Nk"] = Nk
    cfg["kxlist"] = np.linspace(kx_min,kx_max,Nk,endpoint = False)
    cfg["kylist"] = np.linspace(ky_min,ky_max,Nk,endpoint = False)
    cfg["Qxlist"] = np.linspace(Qx_min,Qx_max,NQ,endpoint = False)
    cfg["Qylist"] = np.linspace(Qy_min,Qy_max,NQ,endpoint = False)
    return

#for the py file
def SaveConfig(config_file,cfg):
    keys_to_save = ["version", "hamiltonian", "NQ", "Grid", "U", "epsilon","interaction","c_band","v_band","epsy","epsx","n_Fs","RunType","Range"]
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

def QWZHamiltonian(kx,ky,M):
    pauli_mat =[np.array([[0,1],[1,0]]), np.array([[0,-1j],[1j,0]]), np.array([[1,0],[0,-1]])]
    return np.sin(kx)*pauli_mat[0] + np.sin(ky)*pauli_mat[1] + (M+np.cos(kx)+np.cos(ky))*pauli_mat[2]

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

def HexPBCAddInd(k1,k2,nk):
    kx_ind, ky_ind = (a + b for a, b in zip(k1, k2)) #reversed
    kx_ind %= nk
    k_mid = nk // 2  # use integer division
    # how many times does k_x need to wrap?
    wraps = ky_ind // nk
    ky_ind %= nk
    if wraps % 2 != 0:
        if kx_ind >= k_mid:
            kx_ind -= k_mid
        else:
            kx_ind += k_mid
    return kx_ind, ky_ind
def SquarePBCAddInd(idx, delta, N):
    i, j   = idx
    di, dj = delta
    sx = i + di
    sy = j + dj
    nx = sx // N
    ny = sy // N
    ix = sx % N
    jy = sy % N
    return ix, jy, nx, ny

def PBCAddInd(k_wrap,delta,nk,ltype):
    """
    k_wrap: (i,j,nx,ny) representing the coord in the first BZ and how many wraps on each
    delta: just id,jd
    """
    if ltype== "hexagonal":
        return HexPBCAddInd(k_wrap,delta,nk)
    elif ltype == "square":
        i,j,nx,ny = k_wrap
        kx,ky,dnx,dny = SquarePBCAddInd((i,j),delta,nk)
        return (kx,ky,nx+dnx,ny+dny)
    else:
        raise Exception(f"PBC type not configured for this lattice \"{ltype}\"") 

def UnitPhase(z, eps=1e-14):
    a = np.abs(z)
    return z / (a + eps)

def ParallelTransportCylindrical(u, enforce_ky_periodic=True):
    """
    u: shape (Nx, Ny, Norb, Nband), normalized.
    Returns u_g in cylindrical gauge:
      1) ky-parallel transport on i=0 column
      2) optional ky-periodic sewing on i=0 column
      3) x-parallel transport on each ky row
    """
    u_g = np.zeros_like(u, dtype=np.complex128)
    Nx, Ny, Norb, Nband = u.shape

    for band in range(Nband):
        u_g[0, 0, :, band] = u[0, 0, :, band]

        for j in range(Ny - 1):
            z = np.vdot(u_g[0, j, :, band], u[0, j + 1, :, band])   # <j|j+1>
            u_g[0, j + 1, :, band] = np.conj(UnitPhase(z)) * u[0, j + 1, :, band]

        if enforce_ky_periodic:
            z_end = np.vdot(u_g[0, Ny - 1, :, band], u_g[0, 0, :, band])
            theta = np.angle(z_end)

            for j in range(Ny):
                u_g[0, j, :, band] *= np.exp(-1j * j * theta / Ny)

        for j in range(Ny):
            for i in range(Nx - 1):
                z = np.vdot(u_g[i, j, :, band], u[i + 1, j, :, band])   # <i|i+1>
                u_g[i + 1, j, :, band] = np.conj(UnitPhase(z)) * u[i + 1, j, :, band]

    return u_g

def SeamPhaseGx(u_g):
    #convention, use gx[j] for the target j
    """
    u_g: (Nx,Ny,Norb,Nband) after parallel gauge in x
    returns gx: (Ny,Nband,) complex unimodular
    """
    Nx, Ny, _ , Nband= u_g.shape
    gx = np.empty((Ny,Nband), dtype=complex)
    for band in range(Nband):
        for j in range(Ny):
            z = np.vdot(u_g[Nx-1, j,:,band], u_g[0, j,:,band])
            gx[j,band] = z / (np.abs(z) + 1e-14)
    return gx

def SolveH(kx,ky,hcfg):
    kind = hcfg.get("kind",None)
    if kind == "mos2":
        params = hcfg["params"]
        terms = set(map(str.lower, hcfg.get("terms", [])))
        H = TNNMoS_2Hamiltonian(kx,ky,terms,params)
    elif kind == "qwz":
        M = hcfg.get("params").get("M")
        H = QWZHamiltonian(kx,ky,M)
    elif kind =="haldane":
        params= hcfg.get("params")
        M_=params.get("M")
        phi_ = params.get("phi")
        H = HaldaneH(kx,ky,phi=phi_,M=M_)
    elif kind =="qwz analytic":
        M = hcfg.get("params").get("M")
        return GetAnalyticQWZ(kx,ky,M)
    else:
        raise ValueError("Could not parse acceptable Hamiltonian")
    #if we are here we are solving numerically
    val,evec = np.linalg.eigh(H)
    sorted_indices = np.argsort(val)
    sorted_val = val[sorted_indices]
    sorted_evec = evec[:, sorted_indices]
    return sorted_val,sorted_evec

def SolveBzNoninteracting(cfg):
    hcfg = cfg.get("hamiltonian")
    ltype = hcfg["lattice"]
    epsx_factor = cfg.get("epsx",0)
    epsy_factor = cfg.get("epsy",0)
    kx = cfg["kxlist"]
    ky = cfg["kylist"]
    epsx = kx[1]*epsx_factor
    epsy = ky[1]*epsy_factor
    n_points = cfg["Nk"]
    n_bands = cfg["Bands"]
    wf_BZ = np.zeros((n_points,n_points,n_bands,n_bands), dtype=complex) #kx,ky,orbital weight, band
    eval_BZ = np.zeros((n_points,n_points,n_bands))
    for i in range(n_points):
        for j in range(n_points):      
            (val,vec)=SolveH(kx[i]+epsx,ky[j]+epsy,hcfg)
            eval_BZ[i,j] = val
            wf_BZ[i,j] = vec
    if ltype == "square":
        wf_BZ[:,:,:,:] = ParallelTransportCylindrical(wf_BZ[:,:,:,:])
    return eval_BZ, wf_BZ

def GetH2Band(Qx,Qy,ni_wf,ni_evals,cfg,gx):
    U = cfg["U"]
    epsilon = cfg["epsilon"]
    kxlist = cfg["kxlist"]
    kylist = cfg["kylist"]
    Qxlist = cfg["Qxlist"]
    Qylist = cfg["Qylist"]
    Nk = cfg["Nk"]
    range_handle = cfg["Range"]
    interaction = cfg["interaction"]
    params = interaction["params"]
    c_band = cfg["c_band"]
    v_band = cfg["v_band"]
    ltype = cfg.get("hamiltonian").get("lattice")
    if ltype == "hexagonal":
        wk = 2/(np.sqrt(3)*pow(Nk,2)) #2/sqrt3 from triangular lattice
    elif ltype == "square":
        wk = 1/pow(Nk,2)
    else:
        raise ValueError(f"type {ltype} not implemented for BSE kernel")
    if range_handle == "foreward":
        qxi = Qx
        qyi = Qy
    elif range_handle == "center":
        NQ = cfg["NQ"]
        qxi = Qx - NQ//2
        qyi = Qy - NQ//2        
    H = np.zeros((Nk,Nk,Nk,Nk), dtype = complex) #px,py,px',py'
    for px in range(Nk):
        for py in range(Nk):
            Kp = (px,py,0,0)
            Kv = PBCAddInd(Kp,(-qxi,-qyi),Nk,ltype)
            Kc = PBCAddInd(Kp,(qxi,qyi),Nk,ltype)
            for pxp in range(Nk):
                for pyp in range(Nk):
                    Kpp = (pxp,pyp,0,0)
                    Kvp = PBCAddInd(Kpp,(-qxi,-qyi),Nk,ltype)
                    Kcp = PBCAddInd(Kpp,(qxi,qyi),Nk,ltype)
                    if px==pxp and py==pyp:
                        e = ni_evals[Kc[0]][Kc[1]][c_band]- ni_evals[Kv[0]][Kv[1]][v_band]
                    else:
                        e = 0
                    if interaction.get("kind") == "coulomb":
                        V_d_weight = params.get('vd',1)
                        V_x_weight = params.get('vx',1)
                        k_vec = np.array([kxlist[Kv[0]],kylist[Kv[1]]])
                        k_vecp = np.array([kxlist[Kvp[0]],kylist[Kvp[1]]])
                        deltak_vec = k_vec-k_vecp
                        deltak_norm  = np.linalg.norm(deltak_vec)
                        V_d = V_d_weight/(deltak_norm+epsilon)
                        if px==pxp and py==pyp:
                            V_d = 0
                        Q_vec = np.array([Qxlist[Qx],Qylist[Qy]])
                        Q_norm  = np.linalg.norm(Q_vec)
                        V_x = V_x_weight/(Q_norm+epsilon)
                    elif interaction.get("kind") == "constant":
                        V_d = params.get('vd')
                        V_x = params.get('vx')
                    elif interaction.get("kind") == "seperable":
                        V = params.get('vx')
                        #<u_c(p+Q/2)|u_c(p-Q/2)><u_v(p'-Q/2)|u_v(p'+Q/2)>
                        # <u_c(Kc)|u_c(Kv)><u_v(Kv)|u_v(Kc)>
                        uc_c = u_ext(ni_wf,gx,Kc,c_band)
                        uc_v = u_ext(ni_wf,gx,Kv,c_band)
                        uv_vp = u_ext(ni_wf,gx,Kvp,v_band)
                        uv_cp = u_ext(ni_wf,gx,Kcp,v_band)
                        I = U*V*np.vdot(uc_c,uc_v)*np.vdot(uv_vp,uv_cp)
                        H[px,py,pxp,pyp] = e - wk*I
                        continue
                    else:
                        raise ValueError("Could not parse acceptable interaction kind")
                    uc = u_ext(ni_wf,gx,Kc,c_band)
                    uv = u_ext(ni_wf,gx,Kv,v_band)
                    ucp = u_ext(ni_wf,gx,Kcp,c_band)
                    uvp = u_ext(ni_wf,gx,Kvp,v_band)
                    D = U*V_d*np.vdot(uc, ucp) * np.vdot(uvp, uv)
                    X = U*V_x*np.vdot(uc, uv) * np.vdot(uvp, ucp)
                    H[px,py,pxp,pyp] = e - wk*(D-X)
    return H

def u_ext(u, gx, K, band):
    i, j, nx, ny = K
    # cylindrical gauge: seam only in x
    phase = gx[j, band]**nx
    return phase * u[i, j, :, band]

def SolveInteracting(ni_evals,ni_wf,cfg,gx):
    Nk = cfg["Nk"]
    NQ = cfg["NQ"]
    eQ_matrix = np.zeros((NQ,NQ,Nk*Nk))
    f_matrix = np.zeros((NQ, NQ, Nk, Nk, Nk*Nk), dtype=np.complex128)
    for i in range(NQ):#Qx
        for j in range(NQ):#Qy
            H_Q = GetH2Band(i,j,ni_wf,ni_evals,cfg,gx)
            s = H_Q.shape
            H_Q2D = H_Q.reshape(s[0]*s[1], s[2]*s[3]) #p_ij,p'_ij
            e_Q, f_Q = np.linalg.eigh(H_Q2D)
            sorted_indices = np.argsort(e_Q)
            sorted_e_Q = e_Q[sorted_indices]
            sorted_f_Q = f_Q[:, sorted_indices]
            sorted_f_Q_pxpy = sorted_f_Q.reshape(Nk, Nk, Nk*Nk)
            sorted_e_Q = np.real(sorted_e_Q)
            eQ_matrix[i,j]= sorted_e_Q
            f_matrix[i,j] = sorted_f_Q_pxpy
    return(eQ_matrix,f_matrix)
def _SavePerQ(path, i, j, e_all, f_keep, meta=None):
    """Save per-Q results atomically."""
    path = Path(path)
    tmp = path / f"Q_{i:03d}_{j:03d}.tmp.npz"
    out = path / f"Q_{i:03d}_{j:03d}.npz"

    np.savez_compressed(tmp, e_all=e_all, f_keep=f_keep, meta=meta or {})
    os.replace(tmp, out)
    return out

#---Parallel code---#
def SolveOneQTask(Qx, Qy, cfg, ni_wf_path, ni_evals_path, gx_path, outdir):    
    """
    Worker function executed in each process.
    Qx, Qy, cfg_path, ni_wf_path, ni_evals_path, gx_path, outdir
    """
    n_Fs = cfg["n_Fs"]
    # Large arrays: mmap_mode="r" avoids copying; workers open read-only
    ni_wf = np.load(ni_wf_path, mmap_mode="r")
    ni_evals = np.load(ni_evals_path, mmap_mode="r")
    gx = np.load(gx_path, mmap_mode="r")

    # The function GetH2Band must accept these mmap arrays
    try:
        H_Q = GetH2Band(Qx, Qy, ni_wf, ni_evals, cfg, gx)
    except Exception as e:
        # propagate error in a readable way for the driver
        raise RuntimeError(f"GetH2Band failed for Q=({Qx},{Qy}): {e}")

    s = H_Q.shape
    Nk = cfg["Nk"]
    Nk2 = s[0] * s[1]  # should equal Nk*Nk
    H_Q2D = H_Q.reshape(Nk2, Nk2)

    # dense diagonalization; can be swapped later for eigsh/LinearOperator
    e_Q, f_Q = np.linalg.eigh(H_Q2D)
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
def RunAllQParallel(cfg, ni_evals_path, ni_wf_path, gx_path, outdir,n_workers = None, resume=True):
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

    if n_workers is None:
        runtype = cfg.get("RunType", "Local")

        if runtype == "Local":
            # Use almost all local cores
            n_workers = max(1, multiprocessing.cpu_count() - 1)
    
        elif runtype == "Cluster":
            # Prefer scheduler-provided core count
            # SLURM example:
            n_workers = os.environ.get("SLURM_CPUS_PER_TASK")
    
            if n_workers is None:
                # fallback if not running under SLURM
                print("Warning: SLURM_CPUS_PER_TASK not set, falling back to cpu_count()")
                n_workers = multiprocessing.cpu_count()
            else:
                n_workers = int(n_workers)
    
        else:
            raise ValueError(f"Unknown RunType: {runtype}")

    print(f"Running {len(tasks)} Q-points with {n_workers} workers...")

    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = [
            ex.submit(
                SolveOneQTask,
                i, j, cfg,
                str(ni_wf_path), str(ni_evals_path), str(gx_path),
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

def HandlePaths(runtype,timestamp):
    if runtype == "Local":
        tag = ""
        storage_path = Path("Local_Tests")
    elif runtype == "Cluster":
        tag = slurm_tag()
        storage_path = Path("/tobiasholder-storage/ofirbarogel")
    data_folder = Path(f"Data/{timestamp}-{tag}")
    data_path = storage_path / data_folder
    sub_data_path = data_path / Path(f"{timestamp}")
    return data_path, sub_data_path
    
def Main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to config.yml")
    args = ap.parse_args()
    cfg = GetConfig(args.config)
    start = datetime.now()
    # Handle directiries
    timestamp = start.strftime("%Y-%m-%d_%H-%M-%S")
    runtype = cfg["RunType"]
    data_path, sub_data_path = HandlePaths(runtype,timestamp)
    os.makedirs(data_path, exist_ok=True)
    os.makedirs(sub_data_path, exist_ok=True)
    config_file = sub_data_path / "config.yml"
    SaveConfig(config_file,cfg)
    # --- compute noninteracting pieces
    ni_evals, ni_wf = SolveBzNoninteracting(cfg)
    gx = SeamPhaseGx(ni_wf)

    # directory layout
    perq_dir = Path(data_path) / "perQ"
    inputs_dir = Path(data_path) / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    perq_dir.mkdir(parents=True, exist_ok=True)

    # Save large arrays to disk once (workers will mmap them)
    ni_wf_path = inputs_dir / "ni_wf.npy"
    ni_evals_path = inputs_dir / "ni_evals.npy"
    gx_path = inputs_dir / "gx.npy"
    np.save(ni_wf_path, ni_wf)
    np.save(ni_evals_path, ni_evals)
    np.save(gx_path, gx)

    # Run the parallel Q pipeline (computes and saves per-Q npz files)
    RunAllQParallel(cfg, str(ni_evals_path), str(ni_wf_path), str(gx_path),
                       perq_dir, resume=True)

    # Stitch the outputs you want:
    NQ = cfg["NQ"]
    n_Fs = cfg["n_Fs"]
    Nk = cfg["Nk"]
    Nk2 = Nk * Nk
    # 1) full eigenvalue stitching (cheapish)
    evals_all_path = sub_data_path / "Exciton_Energys.npy"
    DataManager.StitchEvals(perq_dir, evals_all_path, NQ)

    # 2) eigenvectors only up to n_Fs (only if n_Fs>0)
    evecs_keep_path = sub_data_path / "F_keep.npy"
    DataManager.StitchEvecs(perq_dir, evecs_keep_path, NQ, n_Fs) #Saves F[NQ,NQ,Nk^2,n_Fs]

    # Now downstream code can use these .npy or load per-Q files on demand.
    if n_Fs and n_Fs > 0:
        # open memmap for eigenvectors: shape (NQ, NQ, Nk2, n_Fs)
        Fmem = np.load(evecs_keep_path, mmap_mode="r")
        for n in range(n_Fs):
            # reconstruct Fn in original shape expected by ExcitonWF
            Fn_flat = Fmem[:, :, :, n]  # shape (NQ, NQ, Nk2)
            Fn = Fn_flat.reshape(NQ, NQ, Nk, Nk)
            v = cfg.get("v_band")
            c = cfg.get("c_band")
            UnBC, Cn = ExcitonWF.GetExcitonBC(Fn, ni_wf, v, c, cfg.get("hamiltonian").get("lattice"), gx, cfg["Range"])
            UnProb = ExcitonWF.GetExcitonProbability(Fn, ni_wf, v, c, cfg.get("hamiltonian").get("lattice"), gx, cfg["Range"])
            SaveData(Fn, f"F[{n}]", sub_data_path)
            SaveData(UnBC, f"U[{n}]BC_C_{Cn}", sub_data_path)
            SaveData(UnProb, f"U[{n}]_Prob", sub_data_path)
    end = datetime.now()
    elapsed = end-start
    (data_path / "Runtime.txt").write_text(f"{elapsed}\n")
if __name__ == "__main__":
    Main()