"""spectrogram.py

OLIVER MELCHERT

Module containing function definitions for calculating a spectrograms and for
producing a simple figure.

CONTENT:
    - spectrogram
    - plot_spectrogram
"""

# -- USED LIBRARIES
import numpy as np
import numpy.fft as nfft
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as col


# -- CONVENIENT ABBREVIATIONS
FTFREQ = nfft.fftfreq
FT = nfft.ifft
IFT = nfft.fft
SHIFT = nfft.ifftshift


def spectrogram(t, ut, t_lim=None, Nt=1000, s0=20.0):
    """Compute spectrogram.

    Computes a spectrogram for a time-domain input signal, using a short time
    Fourier transform [1]. Implements a Gaussian window function for localizing
    the signal in time. Interpretation of the resulting time-frequency
    distribution for a supercontinuum generated in a photonic crystal fiber is
    detail in Ref. [2].

    Note:
        - Input signal is expected to be a complex-valued envelope resulting
          from a pulse propagation simulation in terms of a nonlinear
          Schrödinger type equation.
        - Gaussian window function is hard-coded.

    Refs:
        [1] L. Cohen, "Time-frequency distributions – A review," Proc. IEEE 77,
          941 (1989), https://doi.org/10.1109/5.30749.

        [2] J. M. Dudley et al., "Cross-correlation frequency resolved optical
          gating analysis of broadband continuum generation in photonic crystal
          fiber: simulations and experiments," Opt. Express. 10, 1215 (2002),
          https://doi.org/10.1364/OE.10.001215.

    Args:
        t (:obj:`numpy.array`, 1-dim):
              Temporal grid.
        ut (:obj:`numpy-array`, 1-dim):
              Time-domain representation of the field.
        t_lim (:obj:`list`):
              Delay time bounds for temporal axis considered for constructing
              the spectrogram (tMin, tMax), default is (min(t),max(t)).
        Nt (:obj:`int`):
              Number of delay times samples in [tMin, tMax], used for signal
              localization (default: Nt=1000).
        s0 (:obj:`float`):
              Root-mean-square width of Gaussian function used for signal
              localization (default: s0=20.0).

    Returns:
        :obj:`list`: (t_spec, w_spec, P_tw), where `t_seq`
        (:obj:`numpy.ndarray`, 1-dim) are delay times, `w`
        (:obj:`numpy.ndarray`, 1-dim) are angular frequencies, and `P_tw`
        (:obj:`numpy.ndarray`, 2-dim) is the spectrogram.
    """
    
    if t_lim == None:
        t_min, t_max = np.min(t), np.max(t)
    else:
        t_min, t_max = t_lim
    # -- DELAY TIMES
    t_seq = np.linspace(t_min, t_max, Nt)
    w = nfft.ifftshift(nfft.fftfreq(t.size,d=t[1]-t[0])*2*np.pi)
    # -- WINDOW FUNCTION
    h = lambda t: np.exp(-(t ** 2) / 2 / s0 / s0) / np.sqrt(2.0 * np.pi * s0 * s0)
    # -- COMPUTE TIME-FREQUENCY RESOLVED CONTENT OF INPUT FIELD
    P = np.abs(FT(h(t - t_seq[:, np.newaxis]) * ut[np.newaxis, :], axis=-1)) ** 2
    return t_seq, w, np.swapaxes(SHIFT(P, axes=-1), 0, 1)


def plot_spectrogram(z_pos, t_delay, w_opt, P_tw, material=None, t_lim = None, w_lim = None, o_name = None,path=""):
    r"""Generate a figure of a spectrogram.

    Generate figure showing the intensity normalized spectrogram.  Scales the
    spectrogram data so that maximum intensity per time and frequency is unity.

    Args:
        t_delay (:obj:`numpy.ndarray`, 1-dim): Delay time grid.
        w_opt (:obj:`numpy.ndarray`, 1-dim): Angular-frequency grid.
        P_tw (:obj:`numpy.ndarray`, 2-dim): Spectrogram data.
    """
    
    if t_lim == None:
        t_min, t_max = t_delay[0], t_delay[-1]
    else:
        t_min, t_max = t_lim

    if w_lim == None:
        w_min, w_max = w_opt[0], w_opt[-1]
    else:
        w_min, w_max = w_lim

    if material:
        resonance=material.expected_resonance()[1][0]
        ZDW=material.ZDW()
        beta2=material.beta[2]
        beta3=material.beta[3]
    
    f, ax1 = plt.subplots(1, 1, sharey=True, figsize=(7,4))
    plt.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.78)
    cmap = mpl.cm.get_cmap("jet")

    #plot GV-parabola
    if material:
        parabel=lambda om,z: (beta2*om+beta3/2*om**2)*z
        ax1.plot(parabel(w_opt,z_pos),w_opt,"w-",alpha=0.5)
        ax1.axhline(resonance,color='white', lw=1, dashes=[3,3])
        ax1.axhline(ZDW,color='white', lw=1,alpha=0.5, dashes=[3,3])



    def _setColorbar(im, refPos):
        """colorbar helper"""
        x0, y0, w, h = refPos.x0, refPos.y0, refPos.width, refPos.height
        cax = f.add_axes([x0+0.35*w, y0+1.02*h, 0.65*w, 0.04*h])
        #cax = f.add_axes([x0, y0 + 1.02 * h, w, 0.05 * h])
        cbar = f.colorbar(im, cax=cax, orientation="horizontal")
        cbar.set_ticks([1e-6,1e-3,1e0])
        cbar.ax.tick_params(
            color="k",
            labelcolor="k",
            bottom=False,
            direction="out",
            labelbottom=False,
            labeltop=True,
            top=True,
            size=4,
            pad=0,
        )

        cbar.ax.tick_params(which="minor", bottom=False, top=False)
        return cbar

    _truncate = lambda I: np.where(I>I.max()*1e-6, I ,  I.max()*1e-6)

    I = _truncate(P_tw[:-1, :-1] / P_tw.max())
    im1 = ax1.pcolorfast(
        t_delay,
        w_opt,
        I,
        norm=col.LogNorm(vmin=1e-6 * I.max(), vmax=I.max()),
        cmap=cmap,
    )
    
    cbar1 = _setColorbar(im1, ax1.get_position())
    z_=int(z_pos/material.soliton_period())
    cbar1.ax.text(1e-9, 1e-4,r"$P_S(\tau, \Omega; z={z_})$".format(z_=z_),color='k',fontsize=18)#[norm.]
    #cbar1.ax.set_title(r"$P_S(\tau, \Omega; z)$", color="k", y=3.5)

    ax1.set_xlim(t_min, t_max)
    ax1.set_ylim(w_min, w_max)
    ax1.tick_params(axis="y", length=2.0, direction="out")
    ax1.tick_params(axis="x", length=2.0, direction="out")
    ax1.set_xlabel(r"Time $\tau$ [ps]")
    ax1.set_ylabel(r"Detuning $\Omega$ [rad/ps]")

    ax1.text(0.02, 0.02, r'$z/L_{sol} = %3.2lf$'%(z_pos/material.soliton_period()), horizontalalignment='left', color='white',
                            verticalalignment='bottom', transform=ax1.transAxes)

    if o_name:
        plt.savefig(path+o_name + ".svg")
        plt.close()
    else:
        plt.show()
        

