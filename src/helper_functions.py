""" helper_functions.py

OLIVER MELCHERT, 01.06.2020

module implementing various helper functions for exercise 04 Computational Photonics"
Leibniz University Hannover in summer term 2017

further helperfunctions added by NIKLAS BAHR during Master's thesis in 2022/2023

CONTENT:
    - energyTD_Z
    - energyFD_Z
    - energy
    - dispersionLength
    - nonlinearLength
    - solitonOrder
    - idxAt
    - flatten
"""

# -- USED LIBRARIES
import numpy as np
import numpy.fft as nfft


FT = nfft.ifft

def energyTD_Z(z,t,Azt,z_):
    """
    t-integrate intensity of given measurement at propagation distance z_ in Time-Domain.
    """
    #z_skip=len(z)//len(Azt)
    zx=idxAt(z,z_)#//z_skip
    return [z_, np.sum(np.abs(Azt[zx])**2)]


def energyFD_Z(z,t,Iw,z_):
    """
    t-integrate intensity of given measurement at propagation distance z_ in Frequency-Domain.
    """
    z_skip=len(z)//len(Iw)
    #w = nfft.ifftshift(nfft.fftfreq(t.size,d=t[1]-t[0])*2*np.pi)
    #Iw = np.abs(nfft.ifftshift(FT(Azt, axis=-1),axes=-1))**2
    zx=idxAt(z,z_)//z_skip
    return np.sum(Iw[zx]) 

def  energy(t,A):
    """Pulse energy

    Function calculating the energy for the time-domain pulse envelope. This
    can be used to monitor energy conservation for the nonlinear Schroedinger
    equation [1].

    Args:
        t (array): time axis
        A (array): time domain pulse profile

    Returns:
        E (float): energy of the pulse envelope

    Refs:
        [1] Split-Step Methods for the Solution of the Nonlinear
            Schroedinger Equation
            J.A.C. Weideman and B.M. Herbst
            SIAM J. Math. Num. Analysis, 23 (1986) 485
    """
    return np.trapz(np.abs(A)**2,x=t)


def dispersionLength(t0,beta2):
    """Dispersion length

    Args:
        t0 (float): pulse duration
        beta2 (float): 2nd order dispersion parameter

    Returns:
        LD (float):  dispersion length
    """
    return t0*t0/np.abs(beta2)


def nonlinearLength(gamma,APeak):
    """Nonlinear length

    Args:
        gamma (float): nonlinear parameter
        APeak (float): peak amplitude

    Returns:
        LNL (float): nonlinear length
    """
    return 1./gamma/APeak/APeak


def solitonOrder(t0,APeak,beta2,gamma):
   """Soliton Order

    Args:
        t0 (float): pulse duration
        APeak (float): peak amplitude
        beta2 (float): 2nd order dispersion parameter
        gamma (float): nonlinear parameter

    Returns:
        N (float): soliton order
   """
   return np.sqrt(dispersionLength(t0,beta2)/nonlinearLength(gamma,APeak))


def idxAt(liste,value):
    """
    This helper fuction takes a 1D-Numpy Array and a Value.
    It returns the list-index where value is closest.
    
    input:
        - liste: 1D-Numpy Array
        - value: float/string
    """
    return np.argmin(np.abs(liste-value))


def flatten(w,liste,interval):
    """
    zentrierter Gleitender Durchschnitt
    See also: https://de.wikipedia.org/wiki/Gleitender_Mittelwert
    """
    new=[]
    n=len(liste)
    for i in range(n):
        if (i>=interval) and (i<n-interval):
            new.append(np.mean(liste[i-interval:i+interval+1]))
    return w[interval:-interval], new

# EOF: helper_functions.py
