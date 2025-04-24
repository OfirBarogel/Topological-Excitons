from __future__ import print_function
from pythtb import * # import TB model class
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import FuncFormatter
import os
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import re
import argparse

storage_path = "/tobiasholder-storage/ofirbarogel"
sys_dict = {"Nk":10,"kxlist":np.linspace(-np.pi,np.pi,10),"kylist":np.linspace(-np.pi,np.pi,10),"NQ":19,"Qxlist":np.linspace(-2*np.pi,2*np.pi,19),"Qylist":np.linspace(-2*np.pi,2*np.pi,19),"U":16,"epsilon":1e-6,"M": -1.1, "Bands": 2} #defaults


# # Non interacting picture

## Solve non-interacting BZ

### Hamiltonians
pauli_mat =[np.array([[0,1],[1,0]]), np.array([[0,-1j],[1j,0]]), np.array([[1,0],[0,-1]])]
def ChernHamiltonian(kx,ky):
    M = sys_dict["M"]
    return np.sin(kx)*pauli_mat[0] + np.sin(ky)*pauli_mat[1] + (M+np.cos(kx)+np.cos(ky))*pauli_mat[2]
# 10.1103/PhysRevB.88.085433
def TNNMoS_2Hamiltonian(kx,ky):
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
           2 * u11 * np.cos(4 * alpha))

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
           2 * u22 * np.cos(4 * alpha))

    # Construct the Hamiltonian matrix
    H_0 = np.array([
        [h0, h1, h2],
        [np.conj(h1), h11, h12],
        [np.conj(h2), np.conj(h12), h22]
    ])
    #B= 0.08
    #identity = np.array([[1,0,0],[0,1,0],[0,0,1]])
    #zeeman_block= identity*B
    #Add SOC
    lambda_soc =  0.073
    Lz = np.array([[0, 0, 0],
                   [0, 0, 2j],
                   [0, -2j, 0]])
    
    # SOC matrix
    SOC_block = (lambda_soc/2) * Lz
    
    # Expand to spin basis
    Hamiltonian = np.block([
        [H_0 +SOC_block, np.zeros_like(H_0)],
        [np.zeros_like(H_0), H_0 - SOC_block]
    ])    
    #return Hamiltonian
    return H_0

#for different Hamiltonians change first line in SolveH
def SolveH(kx,ky):
    H = TNNMoS_2Hamiltonian(kx,ky)
    #H = ChernHamiltonian(kx,ky)
    val,evec = np.linalg.eigh(H)
    sorted_indices = np.argsort(val)
    sorted_val = val[sorted_indices]
    sorted_evec = evec[:, sorted_indices]
    return sorted_val,sorted_evec 
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
    return eval_BZ, wf_BZ

## Berry Curvature over BZ
def GetBerryCurvature1band(N,u):
    link_x=np.zeros([N,N], dtype="complex")
    link_y=np.zeros([N,N], dtype="complex")
    BC=np.zeros([N,N], dtype="complex")
    #epsilon = sys_dict["epsilon"]
    for i in range(N):
        for j in range(N):
            link_x[i,j]=(u[i-1,j,:].conj().T @ u[i,j,:])/np.abs(u[i-1,j,:].conj().T @ u[i,j,:])
            link_y[i,j]=(u[i,j-1,:].conj().T @ u[i,j,:])/np.abs(u[i,j-1,:].conj().T @ u[i,j,:])
    for i in range(N):
        for j in range(N):
            BC[i,j]=np.log(link_x[i,j]*link_y[i-1,j]/(link_x[i,j-1]*link_y[i,j]))
    C=np.sum(BC)/(2*1j*np.pi)
    print(C)
    return BC
def GraphBerryCurvature(berry_curvature):
    #plt.imshow(np.imag(berry_curvature), cmap='viridis', vmin=-0.05, vmax=0.05)
    x = sys_dict["kxlist"]
    y = sys_dict["kylist"]
    plt.imshow(np.imag(berry_curvature), cmap='viridis', aspect='auto', vmin=-0.005, vmax=0.005, origin='lower',
               extent=[x[0], x[-1], y[0], y[-1]])
    
    # Apply formatter to x-axis
    plt.gca().xaxis.set_major_formatter(FuncFormatter(PiFormatter))
    plt.gca().yaxis.set_major_formatter(FuncFormatter(PiFormatter))
    plt.title('Berry Curvature')
    plt.xlabel('kx')
    plt.ylabel('ky')
    cbar = plt.colorbar()
    cbar.set_label('Strength')
    plt.show()
def GraphEnergies(evals):
    plt.imshow(evals)
    plt.title('Energy for M')
    plt.xlabel('Nx')
    plt.ylabel('Ny')
    cbar = plt.colorbar()
    cbar.set_label('Field Strength')
    plt.show()
def InterpEquidistant(points , n):
    x = points[:, 0]
    y = points[:, 1]
    
    # Linear length on the line
    distance = np.cumsum(np.sqrt( np.ediff1d(x, to_begin=0)**2 + np.ediff1d(y, to_begin=0)**2 ))
    distance = distance/distance[-1]
    
    fx, fy = interp1d( distance, x ), interp1d( distance, y )
    
    alpha = np.linspace(0, 1, n)
    x_regular, y_regular = fx(alpha), fy(alpha)
    path = np.column_stack((x_regular, y_regular))
    return path

#define high symmetry points
def CalcBandsNi():
    Gamma = np.array([0.0, 0.0])  # Gamma point
    K = np.array([4 * np.pi / 3, 0.0])  # K point
    M = np.array([np.pi, np.pi / np.sqrt(3)])  # M point
    points = np.array([Gamma, K, M, Gamma])
    k_path = InterpEquidistant(points, 300)
    evals = []
    for k in k_path:
        val,vec = SolveH(k[0],k[1])
        evals.append(val)
    # Convert evals to NumPy array for easy plotting
    evals = np.array(evals)  # Shape: (300, num_bands)
    
    # Compute cumulative distances along k-path
    distances = [0.0]  # First point is at distance 0
    for i in range(1, len(k_path)):
        dist = np.linalg.norm(k_path[i] - k_path[i - 1])  # Euclidean distance
        distances.append(distances[-1] + dist)
    
    distances = np.array(distances)  # Convert to NumPy array
    
    # Find tick positions for Γ, K, and M in the distance array
    tick_positions = [0]  # Start with first point (Gamma)
    tick_labels = ["Γ"]  # Gamma point label
    
    cumulative_length = 0
    for i in range(len(points) - 1):
        segment_length = np.linalg.norm(points[i+1] - points[i])
        cumulative_length += segment_length
        tick_positions.append(cumulative_length)
    
    tick_labels.extend(["K", "M", "Γ"])  # Labels for symmetry points
    
    # Plot band structure
    plt.figure(figsize=(6, 5))
    for band in range(evals.shape[1]):  # Loop over all bands
        plt.plot(distances, evals[:, band], color="black")
    
    # Add vertical lines at symmetry points
    for pos in tick_positions:
        plt.axvline(x=pos, color='gray', linestyle='--', linewidth=0.5)
    
    # Set x-axis ticks and labels
    plt.xticks(tick_positions, tick_labels)
    
    # Labels and title
    plt.xlabel("Wavevector k")
    plt.ylabel("Energy (eV)")  # Adjust units if needed
    plt.title("Band Structure")
    
    plt.show()


# ## Check Berry Curvature

# In[85]:

def CalcBcNi():
    n=100
    sys_dict["kxlist"] = np.linspace(0,2*np.pi,n)
    sys_dict["kylist"] = np.linspace(0,4*np.pi/np.sqrt(3),n)
    sys_dict["Nk"] = n
    sys_dict["Bands"] = 3
    evals , evecs = SolveBzNoninteracting()
    #BC_tot  =0
    #for band in range(6):
    #    BC = GetBerryCurvature1band(n, evecs[:,:,:,band])
    #    BC_tot += BC
    BC = GetBerryCurvature1band(n,evecs[:,:,:,2])
    GraphBerryCurvature(BC)

#TODO: fix this so it works on all bands
def GetH2Band(Qx_indx,Qy_indx,wf_ni, ni_evals,half_dQx,half_dQy):
    U = sys_dict["U"]
    epsilon = sys_dict["epsilon"]
    kxlist = sys_dict["kxlist"]
    kylist = sys_dict["kylist"]
    Qxlist = sys_dict["Qxlist"]
    Qylist = sys_dict["Qylist"]
    NQ = sys_dict["NQ"]
    Nk = sys_dict["Nk"]
    #H = np.zeros((NQ,Nk,NQ,Nk), dtype = complex) #H(k,k')
    H = np.zeros((Nk,Nk,Nk,Nk), dtype = complex) #H(k,k')
    for kx in range(Nk):
        for ky in range(Nk):
            for kx_tag in range(Nk):
                for ky_tag in range(Nk):
                    #precalculate the whole modulu buisness
                    kx_Qx = (kx+Qx_indx)%Nk  
                    kx_Qx_tag = (kx_tag+Qx_indx)%Nk
                    ky_Qy = (ky+Qy_indx)%Nk
                    ky_Qy_tag = (ky_tag+Qy_indx)%Nk
                    if kx==kx_tag and ky==ky_tag:
                        e = ni_evals[kx_Qx][ky_Qy][0]- ni_evals[kx][ky][1]
                    else:
                        e = 0
                    V_d = 1/(pow(Nk,2)*(np.sqrt((pow(kxlist[kx-kx_tag],2) + pow(kylist[ky-ky_tag],2))))+epsilon)
                    V_x = 1/(pow(Nk,2)*(np.sqrt(pow(Qxlist[Qx_indx]+half_dQx,2)+pow(Qylist[Qy_indx]+half_dQy,2)))+epsilon)
                    D = U*V_d*np.vdot(wf_ni[kx_Qx,ky_Qy,:,0],wf_ni[kx_Qx_tag,ky_Qy_tag,:,0])*np.vdot(wf_ni[kx_tag,ky_tag,:,1],wf_ni[kx,ky,:,1]) #N=1?
                    X = U*V_x*np.vdot(wf_ni[kx_Qx,ky_Qy,:,1],wf_ni[kx,ky,:,0])*np.vdot(wf_ni[kx_tag,ky_tag,:,0],wf_ni[kx_Qx_tag,ky_Qy_tag,:,1])
                    H[kx,ky,kx_tag,ky_tag] = e - (D-X)
    return H

def GetH(Qx_indx,Qy_indx,wf_ni, ni_evals,half_dQx,half_dQy,v_indx = [0,1],c_indx=[2,3]): #rn manually setting what bands are cond and val but i can also look at all bands onder 0 or something more general
    U = sys_dict["U"]
    epsilon = sys_dict["epsilon"]
    kxlist = sys_dict["kxlist"]
    kylist = sys_dict["kylist"]
    Qxlist = sys_dict["Qxlist"]
    Qylist = sys_dict["Qylist"]
    NQ = sys_dict["NQ"]
    Nk = sys_dict["Nk"]
    return


# In[17]:


def SolveInteracting(ni_evals,ni_wf):
    start_time = time.time()
    NQ = sys_dict["NQ"]
    Qxlist = sys_dict["Qxlist"]
    Qylist = sys_dict["Qylist"]
    dQx = np.abs(Qxlist[1] - Qxlist[0])
    dQy = np.abs(Qylist[1] - Qylist[0])
    Nk = sys_dict["Nk"]
    eQ_matrix = np.zeros((NQ,NQ,Nk*Nk))
    integrated_f_matrix = np.zeros((NQ,NQ,Nk*Nk),dtype=np.complex128)
    for i in range(NQ):#Qx
        for j in range(NQ):#Qy
            H_Q = GetH2Band(i,j,ni_wf,ni_evals,dQx/2,dQy/2)
            H_Q2D = Flatten4D(H_Q)
            e_Q, f_Q = np.linalg.eigh(H_Q2D)
            # Sort the eigenvalues and reorder eigenvectors accordingly
            sorted_indices = np.argsort(e_Q)
            sorted_e_Q = e_Q[sorted_indices]
            sorted_f_Q = f_Q[:, sorted_indices]
            #if np.all(abs(e_Q.imag) < 1e-10):
            sorted_e_Q = np.real(sorted_e_Q)
            #else:
             #   print(f"Complex eigenvalue in interacting picture at Qx = {Qx} Qy = {Qy}")
            eQ_matrix[i][j]= sorted_e_Q 
            integrated_f_matrix[i][j] = np.sum(sorted_f_Q, axis=0) 
    end_time = time.time()
    print(f"time to run {end_time - start_time} sec")
    return(eQ_matrix,integrated_f_matrix)

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
    
def UnFlatten2D(array2D):
    # Reverse the flattening process
    for row in range(imax * jmax):
        for col in range(kmax * lmax):
            # Convert 2D indices (row, col) back to 4D indices (i, j, k, l)
            i = row // jmax
            j = row % jmax
            k = col // lmax
            l = col % lmax
            
            # Assign the value from 2D array to 4D array
            array4D[i, j, k, l] = array2D[row, col]
    return array4D

# In[20]:


def PiFormatter(value, tick_number):#just to be cute
    return f"{np.round(value/np.pi,3)}π"
def Plot3D(eQ_matrix,n):
    #eQ_matrix is the completed exciton eigenenergies grid while n is the number of energy levels we wish to show
    Qxlist = sys_dict["Qxlist"]
    Qylist = sys_dict["Qylist"]
    Nk = sys_dict["Nk"]
    Qx_mesh, Qy_mesh = np.meshgrid(Qxlist,Qylist)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    for i in range(n): #choose bands
        eQ_sheet = eQ_matrix[:,:,i]
        ax.plot_surface(Qx_mesh, Qy_mesh, eQ_sheet, alpha=0.7, label=f"band {i}")
    ax.xaxis.set_major_formatter(FuncFormatter(PiFormatter))
    ax.yaxis.set_major_formatter(FuncFormatter(PiFormatter))
    ax.set_xlabel('Qx')
    ax.set_ylabel('Qy')
    ax.set_zlabel(r'$\epsilon_Q$')
    plt.title("Exciton EigenEnergy")
    output_folder = "Images"
    #output_path = os.path.join(output_folder, f'Nk={Nk},epsilon={epsilon},U={U},M={M}.png')
    output_path = os.path.join(output_folder, f'MoS2 Excitons 3D Nk={Nk}, bands={n}.png')
    plt.savefig(output_path)
    plt.close()
    #plt.show()


# In[30]:

def Plot2D(eigenvalMatrix):
    plt.imshow(eigenvalMatrix)
    plt.title('Energies heatmap')
    plt.xlabel('Qx')
    plt.ylabel('Qy')
    cbar = plt.colorbar()
    cbar.set_label('E')
    plt.savefig(f'2D_Eigenenergies_NQ={NQ}.png')
    plt.show()
    plt.close()


# In[21]:




# In[22]:


#Only for chern insulator rn
def GraphQxConst(eQ_matrix):
    #Qxlist= sys_dict["Qxlist"]
    Qylist = sys_dict["Qylist"]
    Nk= sys_dict["Nk"]
    epsilon = sys_dict["epsilon"]
    U= sys_dict["U"]
    #M = sys_dict["M"]
    rows, cols, n = eQ_matrix.shape
    center_row = rows // 2
    center_plus1 = rows - center_row
    #band = np.zeros(3*(center_plus1))
    count = n
    for i in range(count):
       # line1 = [eQ_matrix[center_row + j-1, center_row + j-1,i] for j in range(center_plus1)]
       # line2 = [eQ_matrix[rows-1, rows - j-2,i] for j in range(center_plus1-1)] #-2 and -1 to avoid using the same point twice
       # line3 = [eQ_matrix[rows - j-1, center_row-1,i] for j in range(center_plus1-1)]
       # band = line1 + line2 + line3
        band = [eQ_matrix[center_row,j,i] for j in range(cols)]
        plt.plot(Qylist,band)
    #xticks_positions = [
    #    0,                      # Start of the first line
    #    len(line1)-1,         # End of the first line
    #    len(line1) + len(line2)-2,  # End of the second line
    #    len(band)-1          # End of the third line
    #]
    #xtick_labels = [r"$\Gamma$", r"M", r"X", r"$\Gamma$"]
    # Set custom xticks
    #plt.xticks(ticks=xticks_positions, labels=xtick_labels)
    plt.xlabel("Qy")
    ax = plt.gca()
    ax.xaxis.set_major_formatter(FuncFormatter(PiFormatter))
    plt.ylabel("energy") #good dealer
    #plt.legend()
    output_folder = "Images"
    #output_path = os.path.join(output_folder, f'Nk={Nk},epsilon={epsilon},U={U},M={M}.png')
    output_path = os.path.join(output_folder, f'MoS2 Nk={Nk},epsilon={epsilon},U={U}.png')
    plt.savefig(output_path)
    plt.close()
    #plt.show()
#GraphQxConst(eQ_matrix)

#run many
#epsilons = [1e-2,1e-3,1e-4,1e-5]
#Us= [8,12,16,20,24]
#for Nk_tag in range(8,16,2):
#    sys_dict["Nk"] = Nk_tag
#    sys_dict["kxlist"] = np.linspace(0,2*np.pi,Nk_tag)
#    sys_dict["kylist"] = np.linspace(0,4*np.pi/sqrt(3),Nk_tag)
#    sys_dict["NQ"] = 2*Nk_tag-1
#    sys_dict["Qxlist"] = np.linspace(0,4*np.pi,2*Nk_tag-1)
#    sys_dict["Qylist"] = np.linspace(0,8*np.pi/sqrt(3),2*Nk_tag-1)
#    for epsilon_tag in epsilons:
#        sys_dict["epsilon"]=epsilon_tag
#        for U_tag in Us:
#            sys_dict["U"]=U_tag
#            ni_evals,ni_wf = SolveBzNoninteracting()
#            eQ_matrix, integrated_f_matrix = SolveInteracting(ni_evals,ni_wf)
#            GraphQxConst(eQ_matrix)


# In[25]:


def add_title_to_image(image_path, title, font_size=40, padding=10):
    image = Image.open(image_path)
    width, height = image.size

    # Height for the title space, which includes padding
    title_height = font_size + 2 * padding

    # Create a new image with extra space for the title (white background)
    new_image = Image.new("RGB", (width, height + title_height), "white")

    # Paste the original image below the title space
    new_image.paste(image, (0, title_height))  # (0, title_height) places the image below the title area

    # Draw the title
    draw = ImageDraw.Draw(new_image)

    try:
        # Attempt to load a font, otherwise fallback to default
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()  # Fallback to default if Arial is missing

    # Use textbbox() to get text size
    bbox = draw.textbbox((0, 0), title, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # Center the text within the title space
    text_x = (width - text_width) // 2  # Horizontally center
    text_y = (title_height - text_height) // 2  # Vertically center with padding

    # Draw the title in black for visibility
    draw.text((text_x, text_y), title, fill="black", font=font)

    return new_image


# In[30]:



image_folder = Path("Images")
gif_folder = Path("Gifs")
def CreateGifs():
    # Here write an array for whatever variable
    Nk_range =  list(range(8,16,2))
    #U_range = list(range(8, 25, 4))
    #epsilon_range = [1e-2,1e-3,1e-4,1e-5]
    # Define constants
    #_N = "14"
    _epsilon = "1e-05"
    _U = "16" 
    _M = "-1.1"
    matching_images=[]
    for Nk in Nk_range:
        file_path = image_folder / f"Nk={Nk},epsilon={_epsilon},U={_U},M={_M}.png"
        if file_path.exists():
            matching_images.append(file_path)
        else:
            print("File not found:", file_path)
    
    processed_images = []
    for img_path in matching_images:
        title = f"N = {img_path.stem.split(',')[0].split('=')[1]}"
        new_img = add_title_to_image(img_path, title)
        processed_images.append(new_img)
    
    if len(processed_images)>0:
        # Open the first image as the base
        #frames = [Image.open(img) for img in processed_images]
        
        # Save as GIF (loop=0 makes it loop indefinitely)
        output_gif_path = os.path.join(gif_folder, f"Changing N, epsilon {_epsilon}, U {_U}.gif")
        #frames[0].save(output_gif_path, save_all=True, append_images=frames[1:], duration=500, loop=0)
        processed_images[0].save(output_gif_path, save_all=True, append_images=processed_images[1:], duration=500, loop=0)
        print("GIF saved as:", output_gif_path)
    else:
        print("No matching images found, GIF not created.")


# TODO:
# more Nk to smooth
# 
# constant interaction
# 
# once we are confident abt the calculations, new models.

# In[57]:


#integrated_f_matrix = np.abs(integrated_f_matrix)
#rows, cols, n = integrated_f_matrix.shape
#center_row = rows // 2
#center_col = cols // 2
#for i in range(2):
#    line1 = [integrated_f_matrix[center_row + j-1, center_row + j-1,i] for j in range(center_plus1)]
#    line2 = [integrated_f_matrix[rows-1, rows - j-2,i] for j in range(center_plus1-1)]
#    line3 = [integrated_f_matrix[rows - j-2, center_row-1,i] for j in range(center_plus1-1)]
#    band = line1 + line2 + line3
#    plt.plot(band, label = f"band {i}")
#xticks_positions = [
#    0,                      # Start of the first line
#    len(line1)-1,         # End of the first line
#    len(line1) + len(line2)-2,  # End of the second line
#    len(band)-1          # End of the third line
#]
#xtick_labels = [r"$\Gamma$", r"M", r"X", r"$\Gamma$"]
## Set custom xticks
#plt.xticks(ticks=xticks_positions, labels=xtick_labels)
#plt.ylabel("integrated F")
#plt.legend()
#plt.show()


# In[76]:


def GraphIntegratedF(F):
    x = sys_dict["Qxlist"]
    y = sys_dict["Qylist"]
    Nk = sys_dict["Nk"]
    plt.imshow(F, aspect='auto', origin='lower',
               extent=[x[0], x[-1], y[0], y[-1]])
    
    # Apply formatter to x-axis
    plt.gca().xaxis.set_major_formatter(FuncFormatter(PiFormatter))
    plt.gca().yaxis.set_major_formatter(FuncFormatter(PiFormatter))
    plt.colorbar(label='F value abs')
    plt.xlabel('Qx')
    plt.ylabel('Qy')
    plt.title('Integrated F (abs)')
    output_folder = "Images"
    #output_path = os.path.join(gif_folder, f'Nk={Nk},epsilon={epsilon},U={U},M={M}.png')
    output_path = os.path.join(output_folder, f'F for 3band MoS2, bottom 2bands, F[1], Nk={Nk}.png')
    plt.savefig(output_path)
    plt.close()
    #plt.show()

#TODO: change all show() to save
if __name__ == "__main__":
    #args
    parser = argparse.ArgumentParser(description="Calculate Excitons")
    parser.add_argument('-Nk', '--Nk', type=int, help="Number of k points in non interacting BZ (NQ=2Nk)", required=True)
    Nk = parser.parse_args().Nk
    #Images and gifs directories
    os.makedirs(gif_folder, exist_ok=True)  # Ensure output folder exists
    os.makedirs(image_folder, exist_ok=True)
    #run single
    Nk = sys_dict["Nk"]
    sys_dict["kxlist"] = np.linspace(0,2*np.pi,Nk)
    sys_dict["kylist"] = np.linspace(0,4*np.pi/np.sqrt(3),Nk)
    NQ = 2*Nk-1
    sys_dict["NQ"] = NQ
    sys_dict["Qxlist"] = np.linspace(0,4*np.pi,NQ)
    sys_dict["Qylist"] = np.linspace(0,8*np.pi/np.sqrt(3),NQ)
    sys_dict["U"] = 16
    sys_dict["Bands"] = 3
    ni_evals,ni_wf = SolveBzNoninteracting()
    eQ_matrix, integrated_f = SolveInteracting(ni_evals,ni_wf)
    Plot3D(eQ_matrix,1)
    GraphQxConst(eQ_matrix)
    #Fs
    abs_integrated_f_matrix = np.abs(integrated_f[:,:,1])
    GraphIntegratedF(abs_integrated_f_matrix)
# # 
