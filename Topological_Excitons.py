import numpy as np
import time
import os
from pathlib import Path
import argparse
from datetime import datetime
import yaml
import ExcitonWF
#ver 0.1.5
def GetConfig(filename):
    with open(filename, 'r') as stream:
        cfg = yaml.load(stream)#,Loader = yaml.UnsafeLoader)
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
    ltype = cfg["hamiltonian"]["lattice"]
    if grid_handle == "extend":
        Nk = 6*NQ
        cfg["k start"] = int(Nk/3)
        if ltype == "square":
            kx_min = ky_min = -2*np.pi
            kx_max = ky_max = 4*np.pi
            Qx_min = Qy_min = 0
            Qx_max = Qy_max =2*np.pi
        elif ltype == "hexagonal":
            raise ValueError("hexagonal extended grid handling not implemented")
    elif grid_handle == "sew":
        Nk = 2*NQ
        cfg["k start"] = 0
        if ltype == "square":
            kx_min = ky_min = -np.pi
            kx_max = ky_max = np.pi
            Qx_min = Qy_min = -np.pi
            Qx_max = Qy_max = np.pi
        elif ltype == "hexagonal":
            raise ValueError("hexagonal extended grid handling not implemented")
    cfg["Nk"] = Nk
    cfg["kxlist"] = np.linspace(kx_min,kx_max,Nk,endpoint = False)
    cfg["kylist"] = np.linspace(ky_min,ky_max,Nk,endpoint = False)
    cfg["Qxlist"] = np.linspace(Qx_min,Qx_max,NQ,endpoint = False)
    cfg["Qylist"] = np.linspace(Qy_min,Qy_max,NQ,endpoint = False)
    return

#for the py file
def SaveConfig(config_file,cfg):
    keys_to_save = ["version", "hamiltonian", "NQ", "Grid", "U", "epsilon","interaction","c_band","v_band","epsy","epsx","n_Fs"]
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
    params = hcfg.get("params", {})
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

def ParallelTransportX(u):
    """
    u_band: shape (Nx, Ny, Norb, Nband), normalized.
    Returns gauged copy u_g with PT along x for each ky line.
    """
    u_g = u.copy()
    Nx, Ny, Norb, Nband = u_g.shape
    for band in range(Nband):
        for j in range(Ny):           # loop over ky lines
            # fix an anchor phase at i=0 (e.g. make first component real)
            comp = np.argmax(np.abs(u_g[0, j, :, band]))
            phase0 = u_g[0, j, comp, band] / (np.abs(u_g[0, j, comp, band]) + 1e-14)
            u_g[0, j, :, band] /= phase0
    
            for i in range(Nx-1):
                z = np.vdot(u_g[i, j,:,band], u_g[i+1, j,:,band])   # <i|i+1>
                u_g[i+1, j, :, band] *= np.conj(UnitPhase(z)) # multiply by e^{-i arg z}

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
        wf_BZ[:,:,:,:] = ParallelTransportX(wf_BZ[:,:,:,:])
    return eval_BZ, wf_BZ

def GetH2Band(Qx,Qy,wf_ni,ni_evals,cfg,gx):
    U = cfg["U"]
    epsilon = cfg["epsilon"]
    kxlist = cfg["kxlist"]
    kylist = cfg["kylist"]
    Qxlist = cfg["Qxlist"]
    Qylist = cfg["Qylist"]
    Nk = cfg["Nk"]
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
        raise ValueError("die in a car crash")
    H = np.zeros((Nk,Nk,Nk,Nk), dtype = complex) #px,py,px',py'
    for px in range(Nk):
        for py in range(Nk):
            Kp = (px,py,0,0)
            Kv = PBCAddInd(Kp,(-Qx,-Qy),Nk,ltype)
            Kc = PBCAddInd(Kp,(Qx,Qy),Nk,ltype)
            for pxp in range(Nk):
                for pyp in range(Nk):
                    Kpp = (pxp,pyp,0,0)
                    Kvp = PBCAddInd(Kpp,(-Qx,-Qy),Nk,ltype)
                    Kcp = PBCAddInd(Kpp,(Qx,Qy),Nk,ltype)
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
                    else:
                        raise ValueError("Could not parse acceptable interaction kind")
                    uc = u_ext(wf_ni,gx,Kc,c_band)
                    uv = u_ext(wf_ni,gx,Kv,v_band)
                    ucp = u_ext(wf_ni,gx,Kcp,c_band)
                    uvp = u_ext(wf_ni,gx,Kvp,v_band)
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
        storage_path = Path("/Local_Tests")
    elif runtype == "Cluster":
        tag = slurm_tag()
        storage_path = Path("/tobiasholder-storage/ofirbarogel")
    data_folder = Path(f"Data/{timestamp}-{tag}")
    data_path = storage_path / data_folder
    sub_data_path = data_path / Path(f"{timestamp}")
    return data_path, sub_data_path
    

if __name__ == "__main__":
    #parse
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
    #GO
    ni_evals,ni_wf = SolveBzNoninteracting(cfg)
    gx = SeamPhaseGx(ni_wf)
    eQ_matrix, f_matrix = SolveInteracting(ni_evals,ni_wf,cfg,gx)
    print(f"F matrix shape:{f_matrix.shape}")
    SaveData(eQ_matrix,"Exciton_Energys", sub_data_path) #[NQ,NQ,Nk*Nk]
    SaveData(f_matrix,"Fs",data_path) #[NQ,NQ,Nk,Nk,Nk*Nk]
    ltype = cfg.get("hamiltonian").get("lattice")
    for n in range(cfg.get("n_Fs",0)):
        Fn = f_matrix[:, :, :,:, n]
        v= cfg.get("v_band")
        c = cfg.get("c_band")
        UnBC,Cn = ExcitonWF.GetExcitonBC(Fn,ni_wf,v,c,ltype,gx)
        UnProb = ExcitonWF.GetExcitonProbability(Fn,ni_wf,v,c,ltype,gx)
        SaveData(Fn, f"F[{n}]",sub_data_path)
        SaveData(UnBC, f"U[{n}]BC_C_{Cn}",sub_data_path)
        SaveData(UnProb, f"U[{n}]_Prob",sub_data_path)
    end = datetime.now()
    elapsed = end-start
    (data_path / "Runtime.txt").write_text(f"{elapsed}\n")
