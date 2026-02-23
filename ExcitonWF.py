import numpy as np
from pathlib import Path
import argparse
import Topological_Excitons as TE

def ExcitonVProd(Q,Qp,F_ij,ni_wf,v,c,ltype):
    #Q and Q' are tuples of Q indices, F_ij is F[Qx,Qy,k_ij] k_ij is the reshaped index
    Nflat = F_ij.shape[2]
    Nk = int(np.sqrt(Nflat))
    assert Nk * Nk == Nflat, "shape[2] is not a perfect square!"
    NQ = F_ij.shape[0]
    F = F_ij.reshape(NQ,NQ,Nk,Nk) #F[Qx,Qy,kx,ky]
    val=0
    for kx in range(Nk):
        for ky in range(Nk):
            kxQ,kyQ = TE.PBCAddInd((kx,ky),(2*Q[0],2*Q[1]),Nk,ltype) #k_c = k+Q
            px,py = TE.PBCAddInd((kx,ky),Q,Nk,ltype) #p=k+Q/2
            Fp = F[Q[0],Q[1],px,py]
            u_v = ni_wf[kx,ky,:,v]
            u_c = ni_wf[kxQ,kyQ,:,c] #kx,ky,orbital weight, band
            for kxp in range(Nk):
                for kyp in range(Nk):
                    kxQp,kyQp = TE.PBCAddInd((kxp,kyp),(2*Qp[0],2*Qp[1]),Nk,ltype) #k_c = k+Q
                    pxp,pyp = TE.PBCAddInd((kxp,kyp),Qp,Nk,ltype) #p=k+Q/2
                    Fpp = F[Qp[0],Qp[1],pxp,pyp]
                    u_vp = ni_wf[kxp,kyp,:,v]
                    u_cp = ni_wf[kxQp,kyQp,:,c] #kx,ky,orbital weight, band
                    val+= np.vdot(Fpp,Fp)*np.vdot(u_cp,u_c)*np.vdot(u_v,u_vp)
    return val

def GetExcitonBC(F_ij, ni_wf,v,c,ltype):
    N = F_ij.shape[0] #assumes F_ij.shape[N,N,:]
    link_x=np.zeros([N,N], dtype="complex")
    link_y=np.zeros([N,N], dtype="complex")
    BC=np.zeros([N,N], dtype="float")
    for i in range(N):
        for j in range(N):
            iL,jL = TE.PBCAddInd((i,j),(-1,0),N,ltype)
            iD,jD = TE.PBCAddInd((i,j),(0,-1),N,ltype)
            num = ExcitonVProd((i,j),(iL,jL),F_ij,ni_wf,v,c,ltype)
            link_x[i, j] = num / np.abs(num)
            num = ExcitonVProd((i,j),(iD,jD),F_ij,ni_wf,v,c,ltype)
            link_y[i, j] = num / np.abs(num)
    for i in range(N):
        for j in range(N):
            iL,jL = TE.PBCAddInd((i,j),(-1,0),N,ltype)
            iD,jD = TE.PBCAddInd((i,j),(0,-1),N,ltype)
            val = np.imag(np.log(link_x[i,j]*link_y[iL,jL]/(link_x[iD,jD]*link_y[i,j])))
            BC[i,j] = val
    C=np.sum(BC)/(2*np.pi)
    return BC,C

def GetExcitonProbability(F_ij,ni_wf,v,c,ltype):
    N = F_ij.shape[0]
    P = np.zeros([N,N],dtype = 'f')
    for i in range(N):
        for j in range(N):
            P[i,j] = ExcitonVProd((i,j),(i,j),F_ij,ni_wf,v,c,ltype)
    return P

#p basis
def ExcitonVProd_PBasis(Q,Qp,F,ni_wf,v,c,ltype):
    ov = 0.0 + 0.0j
    for px in range(nk):
        for py in range(nk):
            # map p -> kv, kc for Q
            k_vx, k_vy = TE.PBCAddInd((px,py), (-Q[0], -Q[1]), nk, ltype)
            k_cx, k_cy = TE.PBCAddInd((px,py), ( Q[0],  Q[1]), nk, ltype)
            u_v_p = ni_wf[k_vx, k_vy,:,v]   # shape (ncomp,)
            u_c_p = ni_wf[k_cx, k_cy,:,c]

            Fp = F[Q[0],Q[1],px,py]

            for pxp in range(nk):
                for pyp in range(nk):
                    k_vxp, k_vyp = TE.PBCAddInd((pxp,pyp), (-Qp[0], -Qp[1]), nk, ltype)
                    k_cxp, k_cyp = TE.PBCAddInd((pxp,pyp), ( Qp[0],  Qp[1]), nk, ltype)
                    u_v_pp = ni_wf[k_vxp, k_vyp,:,v]
                    u_c_pp = ni_wf[k_cxp, k_cyp,:,c]

                    Fpp = F[Q[0],Q[1],pxp,pyp]

                    # <u_c(kc(p')) | u_c(kc(p))>
                    Scc = np.vdot(u_c_pp, u_c_p)

                    # <u_v(kv(p)) | u_v(kv(p'))>
                    # note order conjugation: np.vdot(a,b) = conj(a)·b
                    Svv = np.vdot(u_v_p, u_v_pp)

                    ov += np.conj(Fpp) * Fp * Scc * Svv

    return ov
    
def ExcitonVProd_SingleSum_PBasis(Q,Qp,Fn,ni_wf,v,c,ltype):
    #accepts Fn[Qx,Qy,px,py] ni_wf[kx,ky,orb,band]
    nk = ni_wf.shape[0]
    nQ = F_ij.shape[0]
    val =0
    for px in range(nk):
        for py in range(nk):
            kc_x,kc_y = TE.PBCAddInd((px,py),Q,nk,ltype) #kc = p+Q/2
            kcp_x,kxp_y = TE.PBCAddInd((px,py),Qp,nk,ltype) #kc' = p+Q'/2
            kv_x,kv_y = TE.PBCAddInd((px,py),(-Q[0],-Q[1]),nk,ltype) #kv = p-Q/2
            kvp_x, kvp_y = TE.PBCAddInd((px,py),(-Qp[0],-Qp[1]),nk,ltype) #kv' = p-Q'/2
            FQ = F[Q[0],Q[1],px,py]
            FQp = F[Q[0],Q[1],px,py]
            uc = ni_wf[kc_x,kc_y,:,c]
            ucp = ni_wf[kcp_x,kcp_y,:,c]
            uv = ni_wf[kv_x,kv_y,:,v]
            uvp = ni_wf[kvp_x,kvp_y,:,v]
            val+= np.vdot(FQp,Fp)*np.vdot(ucp,uc)*np.vdot(uv,uvp)
    return val

def GetExcitonBC_PBasis(F, ni_wf,v,c,ltype,single_sum = False):
    N = F.shape[0] #assumes F_ij.shape[N,N,:]
    link_x=np.zeros([N,N], dtype="complex")
    link_y=np.zeros([N,N], dtype="complex")
    BC=np.zeros([N,N], dtype="float")
    for i in range(N):
        for j in range(N):
            iL,jL = TE.PBCAddInd((i,j),(-1,0),N,ltype)
            iD,jD = TE.PBCAddInd((i,j),(0,-1),N,ltype)
            if single_sum:
                num_x = ExcitonVProd_SingleSum_PBasis((i,j),(iL,jL),F,ni_wf,v,c,ltype)
                num_y = ExcitonVProd_SingleSum_PBasis((i,j),(iD,jD),F,ni_wf,v,c,ltype)
            else:
                num_x = ExcitonVProd_PBasis((i,j),(iL,jL),F,ni_wf,v,c,ltype)
                num_y = ExcitonVProd_PBasis((i,j),(iD,jD),F,ni_wf,v,c,ltype)
            link_x[i, j] = num_x / np.abs(num_x)
            link_y[i, j] = num_y / np.abs(num_y)
    for i in range(N):
        for j in range(N):
            iL,jL = TE.PBCAddInd((i,j),(-1,0),N,ltype)
            iD,jD = TE.PBCAddInd((i,j),(0,-1),N,ltype)
            val = np.imag(np.log(link_x[i,j]*link_y[iL,jL]/(link_x[iD,jD]*link_y[i,j])))
            BC[i,j] = val
    C=np.sum(BC)/(2*np.pi)
    return BC,C
    
def SaveData(data,filename,folder):
    if not filename.endswith('.npy'):
        filename += '.npy'
    full_path = folder / filename
    np.save(full_path, data)

if __name__ == "__main__":
    #Use Exciton WF main to complete calculations from file
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir_parent", required=True, help="Input directory")
    ap.add_argument("--dir_child", required=True, help="Input sub-directory")
    ap.add_argument("--band", required=True,type=int, help="Input number of F bands")    
    args = ap.parse_args()
    storage_path = Path("/tobiasholder-storage/ofirbarogel")
    parent_dir = args.dir_parent
    child_dir = args.dir_child
    bands = args.band
    f_folder = Path(f"Data/{parent_dir}")
    f_file = storage_path/f_folder/Path("Fs.npy")
    config_folder = Path(f"{child_dir}")
    config_name = Path("config.yml")
    config_file = storage_path/f_folder/config_folder/config_name
    cfg = TE.GetConfig(config_file)
    ni_evals,ni_wf = TE.SolveBzNoninteracting(cfg)
    v= cfg.get("v_band")
    c= cfg.get("c_band")
    nk = cfg.get("Nk")
    ltype = cfg["hamiltonian"]["lattice"]
    F= np.load(f_file, mmap_mode='r')
    Fn = F[:,:,:,bands]
    Fnp = TE.FQkToFQpLattice(Fn, nk, ltype)
    UnBC,UnC = GetExcitonBC_PBasis(Fnp,ni_wf,v,c,ltype,True)
    #UnProb = GetExcitonProbability(Fn,ni_wf,v,c,ltype)
    #SaveData(Fn, f"F[{bands}]",config_folder)
    SaveData(UnBC, f"U[{n}]BC_C_{Cn}_single_sum",config_folder)
    #SaveData(UnProb, f"U[{n}]_Prob",config_folder)
        