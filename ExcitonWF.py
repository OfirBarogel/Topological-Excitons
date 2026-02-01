import numpy as np
from pathlib import Path
import argparse
import Topological_Excitons as TE

def ExcitonVProd(Q,Qp,F_ij,ni_wf,v,c,htype):
    #Q and Q' are tuples of Q indices, F_ij is F[Qx,Qy,k_ij] k_ij is the reshaped index
    Nk = ni_wf.shape[0]
    NQ = F_ij.shape[0]
    F = F_ij.reshape(NQ,NQ,Nk,Nk) #F[Qx,Qy,kx,ky]
    val=0
    for kx in range(Nk):
        for ky in range(Nk):
            kxQ,kyQ = TE.PBCAddInd((kx,ky),(2*Q[0],2*Q[1]),Nk,htype) #k_c = k+Q
            px,py = TE.PBCAddInd((kx,ky),Q,Nk,htype) #p=k+Q/2
            Fp = F[Q[0],Q[1],px,py]
            u_v = ni_wf[kx,ky,:,v]
            u_c = ni_wf[kxQ,kyQ,:,c] #kx,ky,orbital weight, band
            for kxp in range(Nk):
                for kyp in range(Nk):
                    kxQp,kyQp = TE.PBCAddInd((kxp,kyp),(2*Qp[0],2*Qp[1]),Nk,htype) #k_c = k+Q
                    pxp,pyp = TE.PBCAddInd((kxp,kyp),Qp,Nk,htype) #p=k+Q/2
                    Fpp = F[Qp[0],Qp[1],pxp,pyp]
                    u_vp = ni_wf[kxp,kyp,:,v]
                    u_cp = ni_wf[kxQp,kyQp,:,c] #kx,ky,orbital weight, band
                    val+= np.vdot(Fpp,Fp)*np.vdot(u_cp,u_c)*np.vdot(u_v,u_vp)
    return val

def GetExcitonBC(F_ij, ni_wf,v,c,htype):
    N = F_ij.shape[0] #assumes F_ij.shape[N,N,:]
    link_x=np.zeros([N,N], dtype="complex")
    link_y=np.zeros([N,N], dtype="complex")
    BC=np.zeros([N,N], dtype="float")
    for i in range(N):
        for j in range(N):
            iL,jL = TE.PBCAddInd((i,j),(-1,0),N,htype)
            iD,jD = TE.PBCAddInd((i,j),(0,-1),N,htype)
            num = ExcitonVProd((i,j),(iL,jL),F_ij,ni_wf,v,c,htype)
            link_x[i, j] = num / np.abs(num)
            num = ExcitonVProd((i,j),(iD,jD),F_ij,ni_wf,v,c,htype)
            link_y[i, j] = num / np.abs(num)
    for i in range(N):
        for j in range(N):
            iL,jL = TE.PBCAddInd((i,j),(-1,0),N,htype)
            iD,jD = TE.PBCAddInd((i,j),(0,-1),N,htype)
            val = np.imag(np.log(link_x[i,j]*link_y[iL,jL]/(link_x[iD,jD]*link_y[i,j])))
            BC[i,j] = val
    C=np.sum(BC)/(2*np.pi)
    return BC,C

def GetExcitonProbability(F_ij,ni_wf,v,c,htype):
    N = F_ij.shape[0]
    P = np.zeros([N,N],dtype = 'f')
    for i in range(N):
        for j in range(N):
            P[i,j] = ExcitonVProd((i,j),(i,j),F_ij,ni_wf,v,c,htype)
    return P
    
def SaveData(data,filename,folder):
    if not filename.endswith('.npy'):
        filename += '.npy'
    full_path = folder / filename
    np.save(full_path, data)

if __name__ == "__main__":
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
    config_folder = Path(f"{child_dir}")
    config_name = Path("config.yml")
    config_file = storage_path/f_folder/config_folder/config_name
    TE.GetConfig(config_file)
    sys_dict = TE.sys_dict
    ni_evals,ni_wf = TE.SolveBzNoninteracting()
    v= sys_dict.get("v_band")
    c= sys_dict.get("c_band")
    for n in range(bands):
        f_name = Path(f"F[{n}].npy")
        f_path = storage_path/f_folder/f_name
        fn = np.load(f_path, mmap_mode='r') #Fn[Qx,Qy,kij]
        UnBC,UnC = GetExcitonBC(fn,ni_wf,v,c)
        UProb = GetExcitonProbability(fn,ni_wf,v,c)
        SaveData(UnBC,f"BC_U{n}_C_{UnC.real:.6g}_{UnC.imag:.6g}j",storage_path/f_folder/config_folder)
        SaveData(UProb,f"U[{n}] Probability.npy",storage_path/f_folder/config_folder)
        