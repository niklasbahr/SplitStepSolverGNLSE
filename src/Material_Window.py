#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NIKLAS BAHR, 11.11.2022

MASTERARBEIT - Zeno-Effect in Nonlinear Optics

THEORETICAL OPTICS AND COMPUTATIONAL PHOTONICS

CONTENT:
    - Material_Properties
    - Computational_Window
"""

# -- USED LIBRARIES
from copy import copy
import matplotlib.pyplot as plt
import numpy as np
import numpy.fft as nfft
from numpy.lib.scimath import sqrt as csqrt
from scipy.special import factorial, erf, erfc

# -- OWN MODULES
from src.helper_functions import idxAt

# -- CONVENIENT ABBREVIATIONS
FT = nfft.ifft
IFT = nfft.fft


class Material_Properties():
    """
    Define material properties for given fiber.
    Define input laser-pulse with following properties:
        
        - alpha:        [linear absorption coefficient (1/m), center frequency (in 10^12 Hz = 1/ps), bandwidth (in 10^12 Hz = 1/ps), profile-mode (flat,erf,symm-flat,symm-erf)]  
        - beta:         taylor expansion of dispersion (in (ps^2/m), (ps^3/m), (ps^4/m), ...)
        - gamma:        nonlinear coefficient (in 1/W/m)
        - TR:           Raman scattering coefficient (in ps)
        - fR:           percentage of contribution of Raman-Effect to Nonlinearity (0,...,1)
        - tau1/2:       Raman-response parameter (in ps)
        
        - P0:           pulse peak power (in W)
        - t0:           pulse width (in ps)
        - omega0:       center frequency (in 1/ps)
        - order:        floating number in case of soliton
        - phase:        initial phase-factor (periodicity depends on window size) (1/ps)
        - shape:        string with decription of pulse-shape

        - delay_phase:  parameter in frequency domain, that implements the temporal delay of a second pulse
        - second_scale: specifies the fraction of the peak power of the second pulse in relation to the main pulse
        
        - noise_seed:   random integer for implementing random quantum noise
    
    Note: Lists shall not be an Numpy-Array, as Json won't function!
    """
    def __init__(self):
        self.alpha=[]
        self.alpha_params=[]
        self.beta=[]
        self.gamma=0
        self.TR=0
        self.tau1 = 0.0122 
        self.tau2 = 0.032
        self.fR = 0.18
        
        self.P0=0
        self.t0=0
        self.omega0=0
        self.order=0
        self.phase=0
        
        self.second_scale=0
        self.delay_phase=0
        
        self.noise_seed=np.random.randint(100000)
        self.shape=""
    
    def define_fiber(self,alpha,beta,gamma,TR=0):
        """
        Define material properties for given fiber.
        Note: Lists shall not be an Numpy-Array, as Json won't function!
        May vary the parameters for absorption profiles to  esnure physical solutions.
        
        input:
         - alpha:  list of [linear absorption coefficient (1/m), center frequency (in 10^12 Hz = 1/ps), bandwidth (in 10^12 Hz = 1/ps),MODE (string)] 
         - beta:   list of [beta0=0,beta1=0,beta2,beta3,...] different orders of dispersion (ps^2/m), (ps^3/m), (ps^4/m), ...
         - gamma:  float with nonlinear coefficient (in 1/w/m)
         - TR:     float of Raman scattering coefficient (in ps)
         
        set:
         - self.alpha: function with frequency dependent absorption 
        """
        if alpha[3]=="flat" or alpha[0]==0 or alpha[2]==0:
            # -- IF BANDWIDTH OR COEFFICIENT IS ZERO, A TRIVIAL FUNCTION IS CREATED
            self.alpha=lambda w: np.where(np.abs(w-alpha[1])<alpha[2],alpha[0],0)
        elif alpha[3]=="erf":
            self.alpha = lambda w: alpha[0]/2*np.where(w<alpha[1],erf(1e1*(w-(alpha[1]-0.8*alpha[2])))+1,erfc(1e1*(w-(0.8*alpha[2]+alpha[1]))))
            # -- DIFFERENT SLOPES AT EDGES
            #self.alpha = lambda w: alpha[0]/2*np.where(w<alpha[1],erf(1e-1*(w-(alpha[1]-0.75*alpha[2])))+1,erfc(1e-1*(w-(0.75*alpha[2]+alpha[1]))))
            #self.alpha = lambda w: alpha[0]/2*np.where(w<alpha[1],erf(1e0*(w-(alpha[1]-0.8*alpha[2])))+1,erfc(1e0*(w-(0.8*alpha[2]+alpha[1]))))
        elif alpha[3]=="symm-flat":
            symm = lambda w: np.logical_or(np.abs(w-alpha[1])<alpha[2],np.abs(w+alpha[1])<alpha[2])#np.abs(w+alpha[1])<alpha[2]#symmetric abbsorbption profile
            self.alpha = lambda w: np.where(symm(w),alpha[0],0)
        elif alpha[3]=="symm-erf":
            shift=0.75; slope = 1e0 #slope == 1e-1 if 1ps, else 10ps
            if alpha[0]==1e10:
                slope=10**(0.2)
            right = lambda w: alpha[0]/2*np.where(w<alpha[1],erf(slope*(w-(alpha[1]-shift*alpha[2])))+1,erfc(slope*(w-(shift*alpha[2]+alpha[1]))))
            left = lambda w: alpha[0]/2*np.where(w<-alpha[1],erf(slope*(w-(-alpha[1]-shift*alpha[2])))+1,erfc(slope*(w-(shift*alpha[2]-alpha[1]))))
            self.alpha = lambda w: right(w)+left(w)
        
        self.alpha_params=list(alpha)
        self.beta=list(beta)
        self.gamma=gamma
        self.TR=TR
        return

    
    def quantum_noise(self,window):
        """
        generates noise by directly sampling in the time domain.
        The underlying noise model assumes complex-valued noise amplitudes
        with normally distributed real and imaginary parts.
        
        input:
        - window:   Computational_Window

        output:
        - noise_t:   (1D numpy-array, cplx floats): instance of time-domain noise
            
        Source: (noise model 01) O. Melchert, A. Demircan, SoftwareX 20 (2022) 101232
        """
        # -- SET NOISE SEED FOR EACH CALL OF THIS FUNCTION. OTHERWISE MAYBE ERRORS WHEN USING LOOPS
        self.noise_seed=np.random.randint(100000)
        
        hBar=6.626e-34/2/np.pi*1e12 # (J ps) reduced Planck constant
        np.random.seed(self.noise_seed)
        N01 = np.random.normal
        t  = window.t
        dt = window.dt #(ps)
        dt *= 1e-12 #(s)  #Skallierungsfehler (Dez 2022)
        
        # -- REPRESENTATIVE ENERGY OF PHOTON IN BIN 
        e0 = hBar*self.omega0                        # (J)
        # -- NOISE SCALING FACTOR ACCOUNTING FOR POWER IN WATTES
        sFac = csqrt(e0/dt/4)      # (sqrt(W) = sqrt(J/s))
        # -- GAUSSIAN NOISE MODEL IN TIME DOMAIN 
        noise_t = sFac*(N01(0,1,size=t.size) + 1j*N01(0,1,size=t.size))
        return noise_t
    
    
    def define_pulse(self,P0=0,t0=0,order=0,omega0=0,shape="soliton"):
        """
        Define input laser-pulse with following properties.
        Requires defined fiber before.
        
        If shape = "soliton":
            - calculate order: if P0 and t0 given
            - calculate P0: if order and t0 given
            - calculate t0: if order and P0 given
        
        input:
         - P0:       pulse peak power (in W)
         - t0:       pulse width (in ps)
         - omega0:   center frequency (in 1/ps= 10^-12 Hz)
         - order:    floating number in case of soliton
         - shape:    string with decription of pulse-shape: soliton, soliton-noise, gaussian, two-pulses
        
        """
        self.P0=P0
        self.t0=t0
        self.order=order
        self.omega0=omega0
        self.shape=shape
        
        if self.shape=="soliton" or self.shape=="gaussian" or self.shape=="soliton-noise" or self.shape[:-1]=="two-pulses" or self.shape =="two-pulses":
            if self.P0 !=0 and self.t0!=0:
                self.order=np.sqrt(self.P0*self.gamma*self.t0**2/np.abs(self.beta[2]))
            elif self.order !=0 and self.t0!=0:
                self.P0=self.order**2*np.abs(self.beta[2])/(self.gamma*self.t0**2)#Klammerfehler 20.01.2022
            elif self.P0 !=0 and self.order!=0:
                self.t0=np.sqrt(self.order**2*np.abs(self.beta[2])/self.P0/self.gamma)
        return
       
        
    def get_pulse(self,window):
        """
        Get an time-domain array of amplitude-distrubition of defined input laser pulse 
        following single pulse shapes are possible: soliton, sech-pulse, gaussian
        for the interaction of two-pulses different modes are implemented. Those are
        customized for a second pulse, frequency matched to the dispersive wave due to
        soliton fission. Dependent on the spectral width of the second pulse, different
        modes are implemented.
        
        input:
         - defined fiber/pulse (self)
         - computational window
         
        output:
         - numpy amplitude-array of length with defined time-steps (len(window.t))
        """
        if (self.shape=="soliton" or self.shape=="sech-pulse"):
            return np.sqrt(self.P0)/np.cosh(window.t/self.t0)
        elif self.shape=="soliton-noise":
            return np.sqrt(self.P0)/np.cosh(window.t/self.t0) + self.quantum_noise(window)
        elif self.shape=="gaussian":
            return np.sqrt(self.P0)*np.exp(-(window.t/(1.76*self.t0))**2)
        elif self.shape =="two-pulses":
            # -- SEE THESIS FOR CASE OF EDW=0.57%
            resonance=self.expected_resonance()[1][0] 
            #INITIAL PULSE
            A_t = np.sqrt(self.P0)/np.cosh(window.t/self.t0)
             #SECOD PULSE
            PW0=np.max(FT(A_t))/self.second_scale
            A_w2= PW0/np.cosh((resonance-window.w)/(0.04*resonance))*np.exp(-1j*window.w*self.delay_phase)#0.04#0.008
            A_t2 = IFT(A_w2, axis=-1)*np.exp(1j*self.phase)
            return A_t+A_t2
        elif self.shape =="two-pulses2":
            # -- SEE THESIS FOR CASE OF EDW=0.10%
            resonance=self.expected_resonance()[1][0]
            #INITIAL PULSE
            A_t = np.sqrt(self.P0)/np.cosh(window.t/self.t0)
            #SECOD PULSE
            PW0=np.max(FT(A_t))/self.second_scale
            A_w2= PW0/np.cosh((resonance-window.w)/(0.02*resonance))*np.exp(-1j*window.w*self.delay_phase)#0.04#0.008
            A_t2 = IFT(A_w2, axis=-1)*np.exp(1j*self.phase)
            return A_t+A_t2
        elif self.shape =="two-pulses3":
            # -- SEE THESIS FOR CASE OF EDW=0.01%
            resonance=self.expected_resonance()[1][0]
            #INITIAL PULSE
            A_t = np.sqrt(self.P0)/np.cosh(window.t/self.t0)
            #SECOD PULSE
            PW0=np.max(FT(A_t))/self.second_scale
            A_w2= PW0/np.cosh((resonance-window.w)/(0.01*resonance))*np.exp(-1j*window.w*self.delay_phase)#0.04#0.008
            A_t2 = IFT(A_w2, axis=-1)*np.exp(1j*self.phase)
            return A_t+A_t2
        
        
    def MI_detuning(self):
        """
        Calculate prediction of expected side band frequency detuning in case of modulation instability (MI).
        Source: Agrawal, Nonlinear Fiber Optics, Chapter 5.1.2. (Optical Solitons)
         
        output:
         - frequency-shift (in 10^12Hz=1/ps)
        """
        if len(self.beta)>2:
            omega_c=np.sqrt(4*self.gamma*self.P0/np.abs(self.beta[2]))
            return omega_c/np.sqrt(2)
        
        
    def expected_resonance(self):
        """
        Calculate (numerical) prediction of expected dispersive-wave frequency  detunning in case higher order dispersion.
        Source: Demircan, 2007, Analysis of the interplay between soliton fission ...
     
        output:
         - frequency-shift (in 10^12Hz=1/ps)
         
        Note: In order to obtain the numerical value, call: material.expected_resonance()[1][0]
        """
        if len(self.beta)>2:
            coeffs=np.asarray([-self.gamma*self.P0,0]+[1/factorial(n) for n in range(2,len(self.beta))])
            beta_arr=np.array(copy(self.beta))
            beta_arr[0]=1; beta_arr[1]=1
            coeffs=np.flip(coeffs*beta_arr)
            phasematching=np.roots(coeffs)
            return "phasematching wave at", phasematching
        
        
    def soliton_period(self):
        """
        Calculates Soliton-Period for soliton of any order
        
        Source: https://www.rp-photonics.com/soliton_period.html (14.12.2022)
        """
        if (("soliton" in self.shape) or self.shape =="two-pulses" or self.shape[:-1] =="two-pulses"):
            return np.pi*self.t0**2/2/np.abs(self.beta[2])
        print("pulse is no soliton. Can not calculate soliton period")
        return None
        
    def LD(self):
        """
        Calculates dispersion legnth (in m) for sech pulse and solitons
        
        Source: Dudley, Genty, and Coen, Rev. Mod. Phys. 78, 1135–1184 (2006)
        """
        if ("soliton" in self.shape or "sech" in self.shape):
            if self.beta[2]!=0:
                return self.t0**2/np.abs(self.beta[2])
            return None
        
    def LNL(self):
        """
        Calculates nonlinear legnth (in m) for sech pulse and solitons
        
        Source: Dudley, Genty, and Coen, Rev. Mod. Phys. 78, 1135–1184 (2006)
        """
        if ("soliton" in self.shape or "sech" in self.shape):
            if self.gamma*self.P0 !=0:
                return 1/(self.gamma*self.P0)
            return None
        
    def ZDW(self):
        """
        Zero-Dispersion-Wavelength
        
        output:
        - omega (in 10^12 Hz=1/ps)
        """
        if self.beta[2]!=0 and self.beta[3]:
            return -self.beta[2]/self.beta[3]
        return None
    
    
    def propagation_constant_plot(self,window,wLim=None,save=False,path=""):
        """
        Creates a plot with characteristics of the frequency-dependent propagation constant beta.
        - propagation constant beta0:      axes[0]
        - relative group delay beta1:      axes[1]
        - group velocity dispersion beta2: axes[2]
        
        phase-matching conditions referring to soliton fission, are highlighted.
            
        Sources: Driben, Yulin, and Efimov, Opt. Express 23, 19112 (2015)
        """
        
        params = {
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        "text.usetex": True,
        'font.family':"Helvetica",
        }
    
        plt.rcParams.update(params)
    
        w=nfft.ifftshift(window.w)
        if not wLim:
            wLim=(np.min(w),np.max(w))
        ZDW=self.ZDW()
        DW=np.real(self.expected_resonance()[1][0])
        DELTA=wLim[1]-ZDW
        wLim=(ZDW-DELTA,wLim[1])
        w1=idxAt(w,wLim[0]);w2=idxAt(w,wLim[1])
        w=w[w1:w2]
        b1=self.beta[1]; b2=self.beta[2]; b3=self.beta[3]; b4=self.beta[4]
        
        PM=lambda om: b2/2*om**2+b3/6*om**3
        GV=lambda om: (b1+b2*om+b3/2*om**2+b4/6*om**3)
        GVD=lambda om: (b2+b3*om+b4/2*om**2)
        
        fig,axes=plt.subplots(3,1,sharex=True,figsize=(10,8))
        axes[0].set_xlim(wLim)
        
        
        
        axes[1].plot(w,GV(w))#beta1
        axes[2].plot(w,GVD(w))#beta2
        
        axes[0].axvline(DW,linestyle="dotted",color="black",alpha=0.4,linewidth=0.5)
        axes[1].axvline(DW,linestyle="dotted",color="black",alpha=0.4,linewidth=0.5)
        axes[2].axvline(DW,linestyle="dotted",color="black",alpha=0.4,linewidth=0.5)
        
        axes[1].axhline(0,linestyle="dashed",color="black",alpha=0.4,linewidth=0.5)
        axes[2].axhline(0,linestyle="dashed",color="black",alpha=0.4,linewidth=0.5)
        
        #highlight anomalous dispersion
        axes[0].axvspan(wLim[0],ZDW, facecolor='lightgrey')
        axes[1].axvspan(wLim[0],ZDW, facecolor='lightgrey')
        axes[2].axvspan(wLim[0],ZDW, facecolor='lightgrey')

        #plots labelling
        axes[0].text(0.01, 0.97, "(a)", horizontalalignment='left', verticalalignment='top', transform=axes[0].transAxes)
        axes[1].text(0.01, 0.97, "(b)", horizontalalignment='left', verticalalignment='top', transform=axes[1].transAxes)
        axes[2].text(0.01, 0.97, "(c)", horizontalalignment='left', verticalalignment='top', transform=axes[2].transAxes)
        
        #Driben
        Lsol=self.soliton_period()
        for N in [-2,-1,1,2]:
            axes[0].axhline(self.gamma*self.P0+2*np.pi*N/Lsol,linewidth=0.5,color="black")
        axes[0].axhline(self.gamma*self.P0,color="red")
        axes[0].plot(w,PM(w))#beta0, als letztes, damit im Vordergrund
        DW=np.real(self.expected_resonance()[1][0])
        axes[0].plot(DW,PM(DW),"ks",markersize=5,alpha=0.5)
        
        axes[2].set_xlabel(r"Detuning $\Omega$ [rad/ps]")
        
        axes[0].set_ylabel(r"$\beta$  [1/m]")
        axes[1].set_ylabel(r"GD  $\beta_1$  [ps/m]")
        axes[2].set_ylabel(r"GVD  $\beta_2$  [ps\textsuperscript{2}/m]")
        fig.align_ylabels()
        
        if save:
            plt.tight_layout()
            plt.savefig(path+"beta_DW{}.svg".format(round(DW,0)))
        return
        
    
    def absorption_profile(self,window,wLim=None,save=False,path="",oName="absorption_profile.pdf"):
        """
        creates plot of absorption-profile in given computational window.
        
        wLim is used to specify the displayed frequency-detuning axis (x-axis)
        by setting save==True, the figure is saved under the referenced path/oName.pdf
        """
        
        params = {
        'figure.figsize': (6,2),
        'axes.labelsize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        "text.usetex": True,
        "font.family": "Helvetica"
        }
    
        plt.rcParams.update(params)
    
        f,a=plt.subplots(figsize=(6,2))
        plt.subplots_adjust(left=0.13, bottom=0.23, right=0.98, top=0.98)
        w = nfft.ifftshift(window.w)
        if wLim==None:
            wLim = (np.min(w),np.max(w))
        a.set_xlim(wLim)
        a.plot(w,self.alpha(w),"k-")
        a.tick_params(axis='both', which='major')
        a.set_xlabel(r"Detuning $\Omega$ [rad/ps]")
        a.set_ylabel(r"loss [m$^{-1}$]")
        if save:
            plt.savefig(path+oName)
        return
    
    def gain_spectrum(self,window,plot=True):
        """
        Calculate gain and loss. Dependent on Fiber/Pulse characteristics.
        
        output:
        - array of len(window.w) gain spectrum substracted by loss of absorption profile
        
        if plot==True: show plot of gain-spectrum
        """
        
        b2=self.beta[2]; gamma=self.gamma; P0=self.P0; alpha = self.alpha
        w=nfft.ifftshift(window.w)
        gain = lambda om: np.imag(csqrt((b2*om**2/2)*(b2*om**2/2+2*gamma*P0)))-alpha(om)/2
        if plot:
            fig,ax = plt.subplots()
            ax.plot(w,gain(w),"k",label="gain")
            plt.legend()
            plt.show()
        return gain(w)


    
class Computational_Window():
    """
    Define properties of Computational Window:
        
        - t_max:   bound for time mesh (in ps)  
        - t_N:     number of sample points: t-axis (shall be a potency of 2)
        - z_max:   upper limit for propagation routine (in m)
        - z_N:     number of sample points: z-axis
        - z_skip:  number of z-steps to keep 
        
        - t:       Numpy-Array with distinct time samples. Calculated from parameters above.
        - z:       Numpy-Array with distinct z samples. Calculated from parameters above.
        - w:       Numpy-Array with distinct frequency samples. Calculated from t.
    
    Important: Lists shall not be an Numpy-Array, as Json won't function!
    """
    
    def __init__(self,t_max=2,t_N=2**12,z_max=0.3,z_N=20000,z_skip=50):
         self.t_max=t_max
         self.t_N=t_N
         self.z_max=z_max        
         self.z_N=z_N
         self.z_skip=z_skip
         
         #time-grid and time-resolution for given parameters
         self.t=np.linspace(-self.t_max, self.t_max, self.t_N, endpoint=False)
         self.dt=self.t[1]-self.t[0] #(in ps)
         
         #propagation-grid and z-resolution for given parameters
         self.z=np.linspace(0, self.z_max, self.z_N, endpoint=True)
         self.dz=self.z[1]-self.z[0] # (in m)
         
         #frequency-grid for given parameters
         self.w= nfft.fftfreq(self.t.size,d=self.dt)*2*np.pi
         self.model=""
    
    def absorb_boundary_TD(self,absorb=1,percent=1,slope=0.3):
        """
        TIME DOMAIN Absorbing Boundaries
        Note: Implemented is flat-absorption profile. In the comments find erf-profile.
        """
        
        TD_absorb=lambda tmax,t: np.where(np.abs(tmax-np.abs(t))/tmax<percent/100,absorb,0)
        
        """
        if absorb == 0:
             TD_absorb=lambda tmax,t:0*t*tmax
        else:   
            TD_absorb=lambda tmax,t: absorb*(erf(slope*(t-percent*tmax))-erf(slope*(t+percent*tmax))+2)
        """
        return TD_absorb(self.t_max,self.t)