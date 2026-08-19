import numpy as np
from pathlib import Path
import argparse
import Topological_Excitons as TE

def CentralDiffPeriodic(arr, axis, step):
    """
    Periodic central difference on a discrete grid.
    arr: array with Q-grid axes leading.
    """
    return (np.roll(arr, -1, axis=axis) - np.roll(arr, 1, axis=axis)) / (2.0 * step)



def BuildExcitonStateOneQ(
    Fn_Q,                 # shape (Np1, Np2)
    ni_wf,            # full Bloch arrays, used by u_ext
    phase_data,
    band_c,
    band_v,
    Q1,
    Q2,
    range_handle="forward"
):
    """
    Build |X(Q)> for one Q-point from the discrete p-grid.

    Returns
    -------
    X : complex array, shape (Np*Np*D,)
        Exciton state in the tensor-product-momentum basis.
        D = Norb * Norb.
    """
    Np1, Np2 = Fn_Q.shape

    Norb = ni_wf.shape[2]
    D = Norb * Norb
    X = np.zeros((Np1, Np2, D), dtype=complex)

    for p1 in range(Np1):
        for p2 in range(Np2):
            if range_handle == "forward":
                q1i = Q1
                q2i = Q2
            elif range_handle == "center":
                NQ = cfg["NQ"]
                q1i = Q1 - NQ//2
                q2i = Q2 - NQ//2        
            Kp = (p1,p2,0,0)
            Kv = TE.PBCAddInd(Kp,(-q1i,-q2i),Nk)
            Kc = TE.PBCAddInd(Kp,(q1i,q2i),Nk)

            uc = TE.u_ext(ni_wf, phase_data, Kc, band_c)
            uv = TE.u_ext(ni_wf, phase_data, Kv, band_v)

            # Tensor product basis vector:
            # (uc ⊗ uv*)_{ab} = uc[a] * conj(uv[b])
            phi = np.kron(uc, np.conj(uv))  # shape (D,)

            X[p1,p2,:] = Fn_Q[p1, p2] * phi
    X = X.reshape(Np1 * Np2 * D)
    return X



def ExcitonProjectorGeometry(
    Fn,                   # shape (nQ1, nQ2, Np1, Np2)
    ni_wf,
    phase_data,
    band_c,
    band_v,
    dQ1=1.0, #b1/NQ
    dQ2=1.0, #b2/NQ
    normalize=True,
    range_handle="forward"
):
    """
    Build the exciton projector P(Q)=|X(Q)><X(Q)| and return the 2x2
    quantum metric g(Q) and BC omega(Q) on the Q-grid.

    Returns
    -------
    g : array, shape (nQ1, nQ2, 2, 2)
    omega : array, shape (nQ1, nQ2, 2, 2)
    X : array, shape (nQ1, nQ2, D)
    P : array, shape (nQ1, nQ2, D, D)
    """
    Fn = np.asarray(Fn, dtype=complex)
    if Fn.ndim != 4:
        raise ValueError("Fn must have shape (nQ1, nQ2, Np1, Np2)")

    nQ1, nQ2, Np1, Np2 = Fn.shape

    # Build all exciton states X(Q1,Q2)
    X_list = []
    for q1 in range(nQ1):
        row = []
        for q2 in range(nQ2):
            X_q = BuildExcitonStateOneQ(
                Fn[q1, q2],
                ni_wf,
                phase_data,
                band_c, band_v,
                q1, q2,
                range_handle=range_handle
            )
            row.append(X_q)
        X_list.append(row)

    X = np.array(X_list, dtype=complex)  # (nQ1, nQ2, D)

    if normalize:
        norm = np.linalg.norm(X, axis=-1, keepdims=True)
        X = np.divide(X, norm, out=np.zeros_like(X), where=(norm > 0))

    # Projector P = |X><X|
    P = X[..., :, None] * np.conj(X[..., None, :])  # (nQ1, nQ2, D, D)

    def trace_matmul(A, B):
        return np.trace(A @ B, axis1=-2, axis2=-1)

    # Finite-difference derivatives of the projector on the Q-grid
    dP_dQ1 = CentralDiffPeriodic(P, axis=0, step=dQ1)
    dP_dQ2 = CentralDiffPeriodic(P, axis=1, step=dQ2)

    dP = [dP_dQ1, dP_dQ2]
    omega = np.zeros((nQ1, nQ2, 2, 2), dtype=float)
    g = np.zeros((nQ1, nQ2, 2, 2), dtype=float)

    for i in range(2):
        for j in range(2):
            g[..., i, j] = 0.5 * np.real(trace_matmul(dP[i], dP[j]))
            omega[..., i, j] = np.real(
                1j * (
                    trace_matmul(P, dP[i] @ dP[j]) -
                    trace_matmul(P, dP[j] @ dP[i])
                )
            )
    # Symmetrize to suppress numerical noise
    g = 0.5 * (g + np.swapaxes(g, -1, -2))
    omega = 0.5 * (omega - np.swapaxes(omega, -1, -2))
    omega_xy = omega[...,0,1]
    return g, omega_xy, X, P

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

def GetExcitonBC(Fn, ni_wf,v,c,ltype,phase_data,range_handle):
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
            KQL = TE.PBCAddInd(KQ,(-1,0),NQ,ltype)
            KQD = TE.PBCAddInd(KQ,(0,-1),NQ,ltype)
            num = ExcitonVProd_PBasis((KQ[0],KQ[1]),(KQL[0],KQL[1]),Fn,ni_wf,v,c,ltype,phase_data,range_handle)
            link_x[i, j] = num / np.abs(num)
            num = ExcitonVProd_PBasis((KQ[0],KQ[1]),(KQD[0],KQD[1]),Fn,ni_wf,v,c,ltype,phase_data,range_handle)
            link_y[i, j] = num / np.abs(num)
    for i in range(NQ):
        for j in range(NQ):
            KQ=(i,j,0,0)
            KQL = TE.PBCAddInd(KQ,(-1,0),NQ,ltype)
            KQD = TE.PBCAddInd(KQ,(0,-1),NQ,ltype)
            val = np.imag(np.log(link_x[KQ[0],KQ[1]]*link_y[KQL[0],KQL[1]]/(link_x[KQD[0],KQD[1]]*link_y[KQ[0],KQ[1]])))
            BC[i,j] = val
    C=np.sum(BC)/(2*np.pi)
    return BC,C

def GetExcitonProbability(Fn,ni_wf,v,c,ltype,phase_data,range_handle):
    """
    accepts Fn[Qx,Qy,px,py] (Qband is selected already)
    """
    NQ = Fn.shape[0]
    P = np.zeros([NQ,NQ],dtype = 'f')
    for i in range(NQ):
        for j in range(NQ):
            P[i,j] = ExcitonVProd_PBasis((i,j),(i,j),Fn,ni_wf,v,c,ltype,phase_data,range_handle)
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

def MainBC(cfg_dir,band, storage_path):
    config_folder = Path(f"{cfg_dir}")
    config_name = Path("config.yml")
    f_file = storage_path/config_folder/Path(f"F[{band}].npy")
    config_file = storage_path/config_folder/config_name
    cfg = TE.GetConfig(config_file)
    v= cfg.get("v_band")
    c= cfg.get("c_band")
    b1 = cfg["rec_vec"][0]
    b2 = cfg["rec_vec"][1]
    NQ = cfg.get("NQ")
    dQ1 = np.linalg.norm(b1)/NQ
    dQ2 = np.linalg.norm(b2)/NQ
    range_handle = cfg["Grid"]["Range"]
    ni_evals,ni_wf = TE.SolveBzNoninteracting(cfg)
    phase_data = TE.SeamHandle(ni_wf,cfg)
    Fn= np.load(f_file, mmap_mode='r')
    metric,BC, exc_state, projector = ExcitonProjectorGeometry(Fn,ni_wf,phase_data,c,v,dQ1=dQ1,dQ2=dQ2, range_handle = range_handle)
    cell_area = abs(np.cross(b1, b2))
    dA = cell_area / (NQ*NQ)
    chern = np.sum(BC) * dA / (2*np.pi)
    #UnBC_test,UnC_test = GetBerryCurvature1band(exc_state)
    #UnBC,UnC = GetExcitonBC(Fn,ni_wf,v,c,gx,range_handle)
    #P = GetExcitonProbability(Fn,ni_wf,v,c,gx,range_handle)
    SaveData(BC, f"Test1_U[{band}]BC_C_{chern}",storage_path/config_folder)
    #SaveData(P, f"ExcWF_U[{band}]_Prob",config_folder)
    return

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
    args = ap.parse_args()
    storage_path = Path("/tobiasholder-storage/ofirbarogel")
    cfg_dir = args.dir
    band = args.band
    #MainBC(cfg_dir,band,storage_path)
    #MainFTrans(cfg_dir,band,storage_path)
        