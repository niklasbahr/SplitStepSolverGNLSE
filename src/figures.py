"""
OLIVER MELCHERT, 01.06.2020

module implementing the figures used during exercise 04 "Computational Photonics"
Leibniz University Hannover in summer term 2017

CONTENT:
    - figure_1a
    - figure_2a (equal)
"""

# -- USED LIBRARIES
import numpy as np
import numpy.fft as nfft
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as col

# -- CONVENIENT ABBREVIATIONS
FT = nfft.ifft
IFT = nfft.fft


def figure_1a(z,t, u, tLim=None ,wLim=None, path=None,oName=None):
    """Plot pulse propagation scene

    Generates a plot showing the z-propagation characteristics of
    the squared magnitude field envelope (left subfigure) and
    the spectral intensity (right subfigure).

    Args:
        z (array): samples along propagation distance
        t (array): time samples
        u (array): time domain field envelope
        tLim (2-tuple): time range in the form (tMin,tMax)
                        (optional, default=None)
        wLim (2-tuple): angular frequency range in the form (wMin,wMax)
                        (optional, default=None)
        oName (str): name of output figure
                        (optional, default: None)
                        
    Information:
        
    Use Line 121 to specify the path where figures shall be saved
    """

    def _setColorbar(im, refPos,scale=[1,1]):
        """colorbar helper"""
        x0, y0, w, h = refPos.x0, refPos.y0, refPos.width, refPos.height
        #cax = f.add_axes([x0, y0+scale[0]*1.02*h, w, scale[1]*0.02*h])
        cax = f.add_axes([x0+0.26*w, y0+1.02*h, 0.74*w, 0.04*h])
        cbar = f.colorbar(im, cax=cax, orientation='horizontal')
        cbar.set_ticks([1e-4,1e-2,1e0])
        #cbar.set_ticklabels([0,0.5,1])
        cbar.ax.tick_params(color='k',
                            labelcolor='k',
                            bottom=False,
                            direction='out',
                            labelbottom=False,
                            labeltop=True,
                            top=True,
                            size=4,
                            pad=0,
                            labelsize=14
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
        'figure.figsize': (8,6),
        'axes.linewidth': 1,
        'lines.linewidth': 1,
        'legend.fontsize': 16,
        'axes.labelsize': 16,
        'font.size': 18,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        "text.usetex": True,
        "font.family": "Helvetica"
        }

    plt.rcParams.update(params)


    w = nfft.ifftshift(nfft.fftfreq(t.size,d=t[1]-t[0])*2*np.pi)

    if tLim==None:
       tLim = (np.min(t),np.max(t))
    if wLim==None:
       wLim = (np.min(w),np.max(w))

    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    plt.subplots_adjust(left=0.1, right=0.9, top=0.85, bottom=0.1)
    cmap=mpl.cm.get_cmap('jet')

    # -- LEFT SUB-FIGURE: TIME-DOMAIN PROPAGATION CHARACTERISTICS
    It = np.abs(u)**2
    It /= np.max(It[0])
    It = _truncate(It)
    im1 = ax1.pcolorfast(t, z, It[:-1,:-1],
                         norm=col.LogNorm(vmin=It.min(),vmax=It.max()),
                         cmap=cmap
                         )
    cbar1 = _setColorbar(im1,ax1.get_position())
    #cbar1.ax.set_title(r"$|A|^2$",color='k')#[norm.]
    cbar1.ax.text(4e-6, 5*1e-3, r"$|A|^2$",color='k')#3e-7 or truncate 1e-5#6e-9
    ax1.xaxis.set_ticks_position('bottom')
    ax1.yaxis.set_ticks_position('left')
    ax1.set_xlim(tLim)
    ax1.set_ylim([0.,z.max()])
    ax1.set_xlabel(r"Time $\tau$ [ps]")
    #ax1.set_ylabel(r"Propagation distance $z$ [m]")
    ax1.set_ylabel(r"Coordinate $z/L_{sol}$ [ - ]")

    # -- RIGHT SUB-FIGURE: ANGULAR FREQUENCY-DOMAIN PROPAGATION CHARACTERISTICS 
    Iw = np.abs(nfft.ifftshift(FT(u, axis=-1),axes=-1))**2
    Iw /= np.max(Iw[0])
    Iw = _truncate(Iw)
    im2 = ax2.pcolorfast(w,z,Iw[:-1,:-1],
                         norm=col.LogNorm(vmin=Iw.min(),vmax=Iw.max()),
                         cmap=cmap
                         )
    cbar2 =_setColorbar(im2,ax2.get_position())
    #cbar2.ax.set_title(r"$|A_\Omega|^2$",color='k')#[norm.]
    cbar2.ax.text(4e-6, 1e-3, r"$|A_\omega|^2$",color='k')#[norm.]#3e-7 for truncate 1e-5,7e-9
    ax2.xaxis.set_ticks_position('bottom')
    ax2.yaxis.set_ticks_position('left')
    #ax2.axvline(1.1683,color='white', lw=1.8,linestyle="dotted")
    ax2.set_xlim(wLim)
    ax2.set_ylim([0.,z.max()])
    
    #ax1.set_ylim([0.,3])
    #ax2.set_ylim([0.,3])
    
    ax2.set_xlabel(r"Detuning $\Omega$ [rad/ps]")
    ax2.tick_params(labelleft=False)


    if oName:
        plt.savefig(path+oName+".pdf",format='pdf',dpi=600)
        #plt.ioff()
    else:
        #plt.savefig(path+"fig/figure.png",dpi=800)
        plt.show()




figure_2a = figure_1a
    
# EOF: figures.py