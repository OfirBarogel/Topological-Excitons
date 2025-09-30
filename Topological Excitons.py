from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import Normalize
import os
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import re
import argparse
from datetime import datetime
import yaml
#ver 0.1.3
def GetConfig(filename):
    global sys_dict
    with open(filename, 'r') as stream:
        sys_dict = yaml.load(stream, Loader=yaml.UnsafeLoader)
    hc = sys_dict["hamiltonian"]
    kind = hc["kind"]
    terms: List[str] = [t for t in hc.get("terms", [])]
    if kind=='qwz':
        sys_dict["Bands"] = 2
    elif kind == 'mos2':
        if any(t in terms for t in ["zeeman", "soc"]):
            sys_dict["Bands"] = 6
        else:
            sys_dict["Bands"] = 3
    Nk = sys_dict.get("Nk")
    sys_dict["kxlist"] = np.linspace(0,2*np.pi,Nk)
    sys_dict["kylist"] = np.linspace(0,4*np.pi/np.sqrt(3),Nk)
    sys_dict["NQ"] = 2*Nk-1
    sys_dict["Qxlist"] = np.linspace(0,4*np.pi,2*Nk-1)
    sys_dict["Qylist"] = np.linspace(0,8*np.pi/np.sqrt(3),2*Nk-1)
    
def SaveConfig(config_file):
    keys_to_save = ["version", "hamiltonian", "Nk", "U", "epsilon","interaction","c_band","v_band"]
    config = {key: sys_dict[key] for key in keys_to_save if key in sys_dict}
    with open(config_file, 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)
def ChernHamiltonian(kx,ky):
    pauli_mat =[np.array([[0,1],[1,0]]), np.array([[0,-1j],[1j,0]]), np.array([[1,0],[0,-1]])]
    M = -1.1
    return np.sin(kx)*pauli_mat[0] + np.sin(ky)*pauli_mat[1] + (M+np.cos(kx)+np.cos(ky))*pauli_mat[2]
# 10.1103/PhysRevB.88.085433
def TNNMoS_2Hamiltonian(kx,ky):
    hcfg   = sys_dict.get("hamiltonian", {})
    terms  = set(map(str.lower, hcfg.get("terms", [])))
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


#for different Hamiltonians change first line in SolveH
def SolveH(kx,ky):
    kind = sys_dict.get("hamiltonian", {}).get("kind",None)
    if kind == "mos2":
        H = TNNMoS_2Hamiltonian(kx,ky)
    elif kind == "qwz":
        H = ChernHamiltonian(kx,ky)
    else:
        raise ValueError("Could not parse acceptable Hamiltonian")
    val,evec = np.linalg.eigh(H) #THIS LINE FUCK THIS LINE
    sorted_indices = np.argsort(val)
    sorted_val = val[sorted_indices]
    sorted_evec = evec[:, sorted_indices]
    return sorted_val,sorted_evec
def FixGauge(wf_BZ):
    Nk = wf_BZ.shape[0]
    n_orb = wf_BZ.shape[2]
    n_bands = wf_BZ.shape[3]
    wf_fixed = np.zeros_like(wf_BZ, dtype=complex)
    for band in range(n_bands):
        # start with (0,0)
        wf_fixed[0,0,:,band] = wf_BZ[0,0,:,band]

        # propagate along ky (j direction)
        for j in range(1, Nk):
            prev = wf_fixed[0, j-1, :, band]
            cur  = wf_BZ[0, j, :, band]
            ov = np.vdot(prev, cur)
            if np.abs(ov) > 1e-12:
                phase = ov/np.abs(ov)
            else:
                phase = 1.0
            wf_fixed[0, j, :, band] = cur * phase.conjugate()

        # now propagate in kx (i direction)
        for i in range(1, Nk):
            # first element in row
            prev = wf_fixed[i-1, 0, :, band]
            cur  = wf_BZ[i, 0, :, band]
            ov = np.vdot(prev, cur)
            if np.abs(ov) > 1e-12:
                phase = ov/np.abs(ov)
            else:
                phase = 1.0
            wf_fixed[i, 0, :, band] = cur * phase.conjugate()

            # then along ky in this row
            for j in range(1, Nk):
                prev = wf_fixed[i, j-1, :, band]
                cur  = wf_BZ[i, j, :, band]
                ov = np.vdot(prev, cur)
                if np.abs(ov) > 1e-12:
                    phase = ov/np.abs(ov)
                else:
                    phase = 1.0
                wf_fixed[i, j, :, band] = cur * phase.conjugate()

    return wf_fixed
def SolveBzNoninteracting():
    kx = sys_dict["kxlist"]
    ky = sys_dict["kylist"]
    n_points = sys_dict["Nk"]
    n_bands = sys_dict["Bands"]
    wf_BZ = np.zeros((n_points,n_points,n_bands,n_bands), dtype=complex) #kx,ky,orbital weight, band
    eval_BZ = np.zeros((n_points,n_points,n_bands))
    for i in range(n_points):
        for j in range(n_points):      
            (val,vec)=SolveH(kx[i],ky[j])
            #for l in range(vec.shape[1]):
            #    phase = np.angle(vec[0,l])
            #    vec = vec * np.exp(-1j*phase)
            eval_BZ[i,j] = val
            wf_BZ[i,j] = vec
    wf_BZ = FixGauge(wf_BZ)
    return eval_BZ, wf_BZ

def GetH2Band(Qx_indx,Qy_indx,wf_ni, ni_evals,half_dQx,half_dQy):
    U = sys_dict["U"]
    epsilon = sys_dict["epsilon"]
    kxlist = sys_dict["kxlist"]
    kylist = sys_dict["kylist"]
    Qxlist = sys_dict["Qxlist"]
    Qylist = sys_dict["Qylist"]
    NQ = sys_dict["NQ"]
    Nk = sys_dict["Nk"]
    interaction = sys_dict["interaction"]
    c_band = sys_dict["c_band"]
    v_band = sys_dict["v_band"]
    H = np.zeros((Nk,Nk,Nk,Nk), dtype = complex)
    for kx in range(Nk):
        for ky in range(Nk):
            for kx_tag in range(Nk):
                for ky_tag in range(Nk):
                    kx_Qx = (kx+Qx_indx)%Nk  
                    kx_Qx_tag = (kx_tag+Qx_indx)%Nk
                    ky_Qy = (ky+Qy_indx)%Nk
                    ky_Qy_tag = (ky_tag+Qy_indx)%Nk
                    if kx==kx_tag and ky==ky_tag:
                        e = ni_evals[kx_Qx][ky_Qy][0]- ni_evals[kx][ky][c_band]
                    else:
                        e = 0
                    if interaction.get("kind") == "coulomb":
                        V_d = 1/(pow(Nk,2)*(np.sqrt((pow(kxlist[kx-kx_tag],2) + pow(kylist[ky-ky_tag],2))))+epsilon)
                        V_x = 1/(pow(Nk,2)*(np.sqrt(pow(Qxlist[Qx_indx]+half_dQx,2)+pow(Qylist[Qy_indx]+half_dQy,2)))+epsilon)
                    elif interaction.get("kind") == "constant":
                        params = interaction.get("params")
                        V_d = params.get('vd')
                        V_x = params.get('vx')
                    else:
                        raise ValueError("Could not parse acceptable interaction kind")
                    D = U*V_d*np.vdot(wf_ni[kx_Qx,ky_Qy,:,v_band],wf_ni[kx_Qx_tag,ky_Qy_tag,:,v_band])*np.vdot(wf_ni[kx_tag,ky_tag,:,c_band],wf_ni[kx,ky,:,c_band])
                    X = U*V_x*np.vdot(wf_ni[kx_Qx,ky_Qy,:,c_band],wf_ni[kx,ky,:,v_band])*np.vdot(wf_ni[kx_tag,ky_tag,:,v_band],wf_ni[kx_Qx_tag,ky_Qy_tag,:,c_band])
                    H[kx,ky,kx_tag,ky_tag] = e - (D-X)
    return H

def SolveInteracting(ni_evals,ni_wf):
    start_time = time.time()
    NQ = sys_dict["NQ"]
    Qxlist = sys_dict["Qxlist"]
    Qylist = sys_dict["Qylist"]
    dQx = np.abs(Qxlist[1] - Qxlist[0])
    dQy = np.abs(Qylist[1] - Qylist[0])
    Nk = sys_dict["Nk"]
    eQ_matrix = np.zeros((NQ,NQ,Nk*Nk))
    f_matrix = np.zeros((NQ,NQ,Nk*Nk,Nk*Nk),dtype=np.complex128)
    for i in range(NQ):#Qx
        for j in range(NQ):#Qy
            H_Q = GetH2Band(i,j,ni_wf,ni_evals,dQx/2,dQy/2)
            H_Q2D = Flatten4D(H_Q)
            e_Q, f_Q = np.linalg.eigh(H_Q2D)
            # Sort the eigenvalues and reorder eigenvectors accordingly
            sorted_indices = np.argsort(e_Q)
            sorted_e_Q = e_Q[sorted_indices]
            sorted_f_Q = f_Q[:, sorted_indices]
            sorted_e_Q = np.real(sorted_e_Q)
            eQ_matrix[i,j]= sorted_e_Q
            f_matrix[i,j] = sorted_f_Q
    end_time = time.time()
    print(f"time to finish {end_time - start_time}")
    return(eQ_matrix,f_matrix)


def Flatten4D(array4D):
    Nk = sys_dict["Nk"]
    imax,jmax,kmax,lmax = Nk,Nk,Nk,Nk #this can be made more generic but im trying to minmax efficiency a little bit
    # Flatten the 4D array to a 2D array using explicit index mapping
    array2D = np.empty((imax * jmax, kmax * lmax), dtype=complex)
    for i in range(imax):
        for j in range(jmax):
            for k in range(kmax):
                for l in range(lmax):
                    # Map 4D indices to 2D indices
                    row = i * jmax + j
                    col = k * lmax + l
                    array2D[row, col] = array4D[i, j, k, l]
    return array2D
    
def SaveData(data,filename,folder):
    if not filename.endswith('.npy'):
        filename += '.npy'
    full_path = folder / filename
    np.save(full_path, data)



if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to config.yml")
    args = ap.parse_args()
    GetConfig(args.config)
    now = datetime.now()
    # Format as string
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    #directories
    data_folder = Path(f"Data/{timestamp}")
    storage_path = Path("/tobiasholder-storage/ofirbarogel")
    data_path = storage_path / data_folder
    os.makedirs(data_path, exist_ok=True)
    config_file = data_path / "config.yml"
    SaveConfig(config_file)
    ni_evals,ni_wf = SolveBzNoninteracting()
    eQ_matrix, f_matrix = SolveInteracting(ni_evals,ni_wf)
    integrated_f = np.sum(f_matrix,axis=2)
    #Fs
    phase_integrated_f_matrix = np.angle(integrated_f[:,:,0])
    #Savedata
    SaveData(integrated_f,"Integrated_Fs", data_path) #[NQ,NQ,Nk*Nk]
    SaveData(eQ_matrix,"Exciton_Energys", data_path) #[NQ,NQ,Nk*Nk]
    SaveData(f_matrix,"Fs",data_path) #[NQ,NQ,Nk*Nk,Nk*Nk]
# # 
