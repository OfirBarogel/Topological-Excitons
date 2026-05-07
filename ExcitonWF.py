import numpy as np
from pathlib import Path
import argparse
import Topological_Excitons as TE

def GetExcitonBC(Fn, ni_wf,v,c,gx,range_handle):
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
            num = ExcitonVProd_PBasis((KQ[0],KQ[1]),(KQL[0],KQL[1]),Fn,ni_wf,v,c,gx,range_handle)
            link_x[i, j] = num / np.abs(num)
            num = ExcitonVProd_PBasis((KQ[0],KQ[1]),(KQD[0],KQD[1]),Fn,ni_wf,v,c,gx,range_handle)
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

def GetExcitonProbability(Fn,ni_wf,v,c,gx,range_handle):
    """
    accepts Fn[Qx,Qy,px,py] (Qband is selected already)
    """
    NQ = Fn.shape[0]
    P = np.zeros([NQ,NQ],dtype = 'f') #'f'
    for i in range(NQ):
        for j in range(NQ):
            P[i,j] = ExcitonVProd_PBasis((i,j),(i,j),Fn,ni_wf,v,c,gx,range_handle)
    return P

def ExcitonVProd_PBasis(Q,Qp,Fn,ni_wf,v,c,gx,range_handle):
    """
    accepts Q and Qp integer touples, Fn[Qx,Qy,px,py] ni_wf[kx,ky,orb,band]
    v,c band indices and gx[ky,band] sewing phase
    """
    nk = ni_wf.shape[0]
    nQ = Fn.shape[0]
    val =0
    if range_handle == "foreward":
        qxi = Q[0]
        qyi = Q[1]
        qxip = Qp[0]
        qyip = Qp[1]          
    elif range_handle == "center":
        qxi = Q[0] - NQ//2
        qyi = Q[1] - NQ//2
        qxip = Qp[0] - NQ//2
        qyip = Qp[1] - NQ//2  
    for px in range(nk):
        for py in range(nk):
            Kp = (px,py,0,0)
            Kc = TE.PBCAddInd(Kp,(qxi,qyi),nk)
            Kv = TE.PBCAddInd(Kp,(-qxi,-qyi),nk)
            Kcp = TE.PBCAddInd(Kp,(qxip,qyip),nk)
            Kvp = TE.PBCAddInd(Kp,(-qxip,-qyip),nk)
            FQ = Fn[Q[0],Q[1],Kp[0],Kp[1]]
            FQp = Fn[Qp[0],Qp[1],Kp[0],Kp[1]]
            uc = TE.u_ext(ni_wf,gx,Kc,c)
            uv = TE.u_ext(ni_wf,gx,Kv,v)
            ucp = TE.u_ext(ni_wf,gx,Kcp,c)
            uvp = TE.u_ext(ni_wf,gx,Kvp,v)
            val+= np.vdot(FQp,FQ)*np.vdot(ucp,uc)*np.vdot(uv,uvp)
    return val
        
def SaveData(data,filename,folder):
    if not filename.endswith('.npy'):
        filename += '.npy'
    full_path = folder / filename
    np.save(full_path, data)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir_parent", required=True, help="Input directory")
    ap.add_argument("--dir_child", required=True, help="Input sub-directory")
    ap.add_argument("--band", required=True,type=int, help="Input desired exciton band")    
    args = ap.parse_args()
    storage_path = Path("/tobiasholder-storage/ofirbarogel")
    parent_dir = args.dir_parent
    child_dir = args.dir_child
    band = args.band
    parent_folder = Path(f"Data/{parent_dir}")
    config_folder = Path(f"{child_dir}")
    config_name = Path("config.yml")
    f_file = storage_path/parent_folder/config_folder/Path(f"F[{band}].npy")
    config_file = storage_path/parent_folder/config_folder/config_name
    cfg = TE.GetConfig(config_file)
    v= cfg.get("v_band")
    c= cfg.get("c_band")
    nk = cfg.get("Nk")
    range_handle = cfg["Range"]
    ni_evals,ni_wf = TE.SolveBzNoninteracting(cfg)
    gx = TE.SeamPhaseGx(ni_wf)
    Fn= np.load(f_file, mmap_mode='r')
    UnBC,UnC = GetExcitonBC(Fn,ni_wf,v,c,gx,range_handle)
    P = GetExcitonProbability(Fn,ni_wf,v,c,gx,range_handle)
    SaveData(UnBC, f"ExcWF_U[{band}]BC_C_{Unc}",config_folder)
    SaveData(P, f"ExcWF_U[{band}]_Prob",config_folder)
        