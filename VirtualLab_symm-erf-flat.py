#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NIKLAS BAHR, 17.07.2023

MASTERARBEIT - Zeno-Effect in Nonlinear Optics

THEORETICAL OPTICS AND COMPUTATIONAL PHOTONICS

CONTENT:
    - VirtualLab
    - Example2_plot (see Ch. 5.7 in thesis)
"""

# -- USED LIBRARIES
import numpy as np
import numpy.fft as nfft
import matplotlib.pyplot as plt
import matplotlib.colors as col
import matplotlib as mpl

# -- OWN MODULES
from src.Material_Window import Material_Properties, Computational_Window
from src.SplitStepSolver import SplitStepSolver
from src.export_import   import save_params, import_params
from src.figures import figure_2a


global figpath, datapath
figpath="fig/"
datapath="res/"

FT = nfft.ifft
IFT = nfft.fft

def VirtualLab():

    # INITIALIZE: Computational window
    window = Computational_Window(t_max=100, t_N=2**14, z_max=25, z_N=10000, z_skip=50)
     
    # INITIALIZE: Fiber and Pulse
    material=Material_Properties()
    
    # DEFINE Fiber Properties
    material.define_fiber(alpha=[1e3,10,1,"symm-erf"], beta=[0,0,-0.03,0.0000], gamma=0.15,TR=0)
    
    # DEFINE Pulse Properties
    material.define_pulse(t0=10, P0=10, omega0=2260, shape='soliton-noise')

    print("soliton period:   ",material.soliton_period())
    print("phase-matching:   ",material.expected_resonance()[1][0])
    
    # INITIALIZE: Solver with window, material, model and absorbing boundaries 
    solver = SplitStepSolver(window,material,'SSFM_NSE_symmetric',False)
    
    # COMPUTE SOLUTION
    z, Azt = solver.compute_model()

    # SAVE RESULTS
    if material.alpha_params[-1]=="symm-erf":
        oName="symmerf"
    elif material.alpha_params[-1]=="symm-flat":
        oName="symmflat"
    else:
        oName="test_data"
    save_params(material, window, Azt, oName= oName+".zip", path= datapath)
    
    # CREATE FIGURE
    figure_2a(z/material.soliton_period(), window.t, Azt, tLim=(-80,80),wLim=(-50,50), path=figpath, oName=oName)

    
    




def Example2_plot():
    

    
    tLim=(-80,80)
    wLim=(-50,50)
    
    def _setColorbar(im, refPos,scale=[1,1]):
        """colorbar helper"""
        x0, y0, w, h = refPos.x0, refPos.y0, refPos.width, refPos.height
        cax = fig.add_axes([x0+0.26*w, y0+1.02*h, 0.74*w, 0.04*h])
        cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
        cbar.set_ticks([1e-4,1e-2,1e0])
     
        cbar.ax.tick_params(color='k',
                            labelcolor='k',
                            bottom=False,
                            direction='out',
                            labelbottom=False,
                            labeltop=True,
                            top=True,
                            size=4,
                            pad=0,
                            labelsize=16
                            )

        cbar.ax.tick_params(which="minor", bottom=False, top=False )
        return cbar

    def _truncate(I):
        """truncate intensity

        fixes python3 matplotlib issue with representing small
        intensities on plots with log-colorscale
        """
        I[I<1e-4]=1e-4
        return I
    
    params = {
        'figure.figsize': (12,6),
        'axes.linewidth': 1,
        'lines.linewidth': 1,
        'legend.fontsize': 16,
        'axes.labelsize': 16,
        'font.size': 16,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        "text.usetex": True,
        "font.family": "Helvetica"
        }

    plt.rcParams.update(params)
    
    fig = plt.figure()
    gridA12 = fig.add_gridspec(nrows=1, ncols=2, left=0.07, right=0.495,wspace=0.15,top=0.85)
    ax1 = fig.add_subplot(gridA12[0])
    ax2 = fig.add_subplot(gridA12[1])
    gridB12 = fig.add_gridspec(nrows=1, ncols=2, left=0.545, right=0.98,wspace=0.15,top=0.85)
    ax3 = fig.add_subplot(gridB12[0])
    ax4 = fig.add_subplot(gridB12[1])
    
    axes = [ax1, ax2, ax3, ax4]
    

    
    def plot_TD_FD(oName,path):
        material, window, z, t, u=import_params(filename=oName+".zip",path=path)
        It = np.abs(u)**2
        It /= np.max(It[0])
        It = _truncate(It)
        
        w = nfft.ifftshift(nfft.fftfreq(t.size,d=t[1]-t[0])*2*np.pi)
        Iw = np.abs(nfft.ifftshift(FT(u, axis=-1),axes=-1))**2
        Iw /= np.max(Iw[0])
        Iw = _truncate(Iw)
        
        return t, It, w, Iw
    
    i=0
    
    lbls=["symm-flat","symm-erf"]
    cmap=mpl.cm.get_cmap('jet')
    oNames=["symmflat","symmerf"]
    for oName in oNames:
        material, window, z, t, u=import_params(filename=oName+".zip",path=datapath)
        
        
        t, It, w, Iw = plot_TD_FD(oName,datapath)
        z/=material.soliton_period()
        w = nfft.ifftshift(nfft.fftfreq(t.size,d=t[1]-t[0])*2*np.pi)
        
        im1= axes[i].pcolorfast(t, z, It[:-1,:-1],
                         norm=col.LogNorm(vmin=It.min(),vmax=It.max()),
                         cmap=cmap
                         ) 

        cbar1 = _setColorbar(im1,axes[i].get_position())
        cbar1.ax.text(4*1e-6, 5*1e-3, r"$|A|^2$",color='k')
        
        im2 = axes[i+1].pcolorfast(w,z,Iw[:-1,:-1],
                         norm=col.LogNorm(vmin=Iw.min(),vmax=Iw.max()),
                         cmap=cmap
                         )
        cbar2 =_setColorbar(im2,axes[i+1].get_position())
        cbar2.ax.text(5*1e-6, 1e-3, r"$|A_\omega|^2$",color='k')

        
        axes[i].set_xlim(tLim)
        axes[i+1].set_xlim(wLim)
     
        
        axes[0].set_ylabel(r"Coordinate $z/L$\textsubscript{sol} [ - ]")
        axes[i].set_xlabel(r"Time $\tau$ [ps]")
        axes[i+1].set_xlabel(r"Detuning $\Omega$ [rad/ps]")
        if i!=0:
            axes[i].tick_params(labelleft=False)
        axes[i+1].tick_params(labelleft=False)

        axes[i].text(0.02, 0.98, lbls[i//2],color="white",bbox=dict(facecolor='black', alpha=0.25, edgecolor='none'),horizontalalignment='left', verticalalignment='top', transform=axes[i].transAxes)
        i+=2
    
    plt.show()
    plt.savefig(figpath+"symm-flat_symm-erf.svg")
    return
   
    

#VirtualLab()
Example2_plot()      