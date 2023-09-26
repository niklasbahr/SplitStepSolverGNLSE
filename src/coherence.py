#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NIKLAS BAHR, 20.01.2023

MASTERARBEIT - Zeno-Effect in Nonlinear Optics

THEORETICAL OPTICS AND COMPUTATIONAL PHOTONICS

CONTENT:
    - firstOrderCoherence
    - import_coherence_measurements_full
    - import_coherence_measurements
    - coherence2
"""

# -- USED LIBRARIES
import os 
import itertools
import numpy as np
import numpy.fft as nfft
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as col

# -- OWN MODULES
from src.export_import import import_params
from src.helper_functions import flatten
from src.figures import figure_2a

# -- CONVENIENT ABBREVIATIONS
FT = nfft.ifft


def firstOrderCoherence(w, Ewz_list):
    """
    interpulse coherence:
    
    Source: GNLStools (https://doi.org/10.1016/j.softx.2022.101232)
    """
    
    Iw_av = np.mean(np.abs(Ewz_list)**2, axis=0)

    nPairs = 0
    XX = np.zeros(len(Ewz_list[0]),dtype=complex)
    for i,j in list(itertools.combinations(range(len(Ewz_list)),2)):
       XX += Ewz_list[i]*np.conj(Ewz_list[j])
       nPairs += 1
    XX = np.abs(XX)/nPairs

    return np.real(XX/Iw_av), Iw_av


def import_coherence_measurements_full(ID,path):
    """
    Import samples of propagation with different noise seeds prepared to calculate the interpulse coherence.
    All samples have to be located in the same "path" and have an unique "ID" at beginning of filename.
  
    returns:
        - z-axis
        - omega-axis
        - lists (*)
        - Iw (first represtentant of the samples, intensity)
        
    lists = [l0, ..., lN] contains the complex amplitudes in frequency domain for each propagation distance.
    l0=[Aw0, ... , AwM] contains a list with all amplitudes at propgation distance z[0], and so on...
    Aw0 denotes the 0th sample in frequency domain (list with len(w) elements)
    N=len(z); M=#samples
    """

    directory = os.fsencode(path)
    INIT=0
    for file in os.listdir(directory):
        fName = os.fsdecode(file)
        print(fName)
        if fName[:len(ID)]==ID:
            material, window, z, t, u = import_params(filename=fName,path=path)
            z/=material.soliton_period()#for normalized plot
            
            if INIT==0:
                # -- KEEP FIRST REPRESENTANT OF NOISE SAMPLES, POSSIBLE TO PLOT THE PROPAGATION SCHEME
                Iw = np.abs(nfft.ifftshift(FT(u, axis=-1),axes=-1))**2
                #figure_2a(z,t, u,tLim=(-10,10),wLim=(-0.25,0.25),path=path,oName="MI_representant")
                # -- INITIALIZE LIST OF LISTS FOR RESULTS
                z_idx=len(u)
                lists = [[] for x in range(z_idx)]
                INIT+=1
                
            for z_val in range(z_idx):
                # -- FILL LIST SEPARATED BY EACH PROPAGATIO STEP 
                lists[z_val].append(nfft.ifftshift(FT(u[z_val])))
             
    w = nfft.ifftshift(nfft.fftfreq(t.size,d=t[1]-t[0])*2*np.pi)*1e-3
    return z, w, lists, Iw

       
def import_coherence_measurements(ID,path):
    """
    Import samples of propagation with different noise seeds prepared to calculate the interpulse coherence.
    Only the last element of the propagation scheme at position z[-1] is considered.
    
    returns:
        - z-axis
        - omega-axis
        - lists (*)
        - Aw0 (first represtentant of the samples, amplitude)
        
    All samples have to be located in the same "path" and have an unique "ID" at beginning of filename.
    """
    
    ### CREATE list with for field  values in Frequency-Domain
    Azw=[] 
    
    directory = os.fsencode(path)
    for file in os.listdir(directory):
        fName = os.fsdecode(file)
        print(fName)
        if fName[:len(ID)]==ID:
            material, window, z, t, u = import_params(filename=fName,path=path)
            Azw.append(nfft.ifftshift(FT(u)))
    
    A0t = material.get_beam(window) 
    A0w=nfft.ifftshift(FT(A0t))  
    w = nfft.ifftshift(nfft.fftfreq(t.size,d=t[1]-t[0])*2*np.pi)*1e-3
  
    return z,w, np.asarray(Azw), A0w



def coherence2(subpath=None, ID='last_'):
    """
    Show coherence of last position for  samples located in path specified below and input-parameter "subpath"
    """
    
    path='/Users/niklas.bahr/Masterarbeit/01_Parameter/MI6/'+subpath+'/'
    z,w, Azw, A0w = import_coherence_measurements(ID,path)
    
    def _truncate(I):
        """truncate intensity

        fixes python3 matplotlib issue with representing small
        intensities on plots with log-colorscale
        """
        I[I<1e-6]=1e-6
        return I
    
    #wLim=(-1.5,2)
    #wLim=(-6,6)
    wLim=(-2,2)
    print("sample number: ",len(Azw))
    coh_w_g, int_Azw = firstOrderCoherence(w, Azw)
    ### PLOT RESULTS
    fig = plt.figure(figsize=(8,5))
    grid = fig.add_gridspec(nrows=3, ncols=1, left=0.15, right=0.95,hspace=0.25,top=0.9,bottom=0.1)
    fig.suptitle('Spectral Coherence at z={_z}'.format(_z=z.max()), fontsize=16)
    
    ax0  = fig.add_subplot(grid[0])
    ax1  = fig.add_subplot(grid[1])
    ax2  = fig.add_subplot(grid[2])
    axes = [ax0,ax1,ax2]
    
    I0w=np.abs(A0w)**2
    I0w /= np.max(I0w)
    int_Azw /= np.max(I0w)
    
    
    #Spectral propagation
    axes[0].plot(w, int_Azw,color="black")
    axes[0].plot(w, I0w,color="lightgrey",linewidth=1)
    axes[0].set_ylim((1e-8,1e1))
    axes[0].set_yscale("log")
    axes[0].tick_params(labelbottom=False)
    axes[0].set_ylabel(r"$|A_\omega|^2$ [norm.]".format(z_=z.max()))
    axes[0].text(0.01, 0.95, "(a)",color="black", horizontalalignment='left', verticalalignment='top', transform=axes[0].transAxes)
    
    
    #Coeherence propagation
    axes[1].plot(w,coh_w_g,color="lightgrey")
    #axes[1].plot(w,Gam,color="lightgrey")
    w,coh_w_g=flatten(w,coh_w_g,10)
    axes[1].plot(w,coh_w_g,linewidth=1,color="black")
    axes[1].tick_params(labelbottom=False)
    axes[1].set_ylabel(r"$|g_{12}(\Omega)|$")
    axes[1].text(0.01, 0.95, "(b)",color="black", horizontalalignment='left', verticalalignment='top', transform=axes[1].transAxes)
    axes[1].set_ylim((0,1))
    Iw=np.abs(Azw)**2
    Iw /= np.max(Iw[0])
    Iw = _truncate(Iw)

    cmap=mpl.cm.get_cmap('jet')
    samples=np.arange(1,len(Iw)+1)
    
    #compare samples at end
    axes[2].pcolorfast(w, samples, Iw, norm=col.LogNorm(vmin=Iw.min(),vmax=Iw.max()), cmap=cmap)
    axes[2].set_ylabel("sample no.")
    axes[2].text(0.01, 0.95, "(c)",color="white", horizontalalignment='left', verticalalignment='top', transform=axes[2].transAxes)
    axes[2].set_xlabel(r"Detuning $\Omega$ [rad/ps]")
    plt.setp(axes, xlim=wLim)
    plt.show()    
    plt.savefig(path+"coh_"+subpath+".png",dpi=600)
    return
