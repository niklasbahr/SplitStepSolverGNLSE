#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NIKLAS BAHR, 03.02.2023

MASTERARBEIT - Zeno-Effect in Nonlinear Optics

THEORETICAL OPTICS AND COMPUTATIONAL PHOTONICS

CONTENT:
    - SplitStepSolver
"""

# -- USED LIBRARIES
import numpy as np
import numpy.fft as nfft
from scipy.special import factorial

# -- OWN MODULES
from src.Material_Window import Material_Properties, Computational_Window

# -- CONVENIENT ABBREVIATIONS
FT = nfft.ifft
IFT = nfft.fft


class SplitStepSolver():
    """
    
    This implementation computes nonlinear pulse propagation
    - for different propagation models
    - for given set of material/window-parameters.
    
    "SSFM_NSE_symmetric":       Nonlinear-Schrödinger-Equation ONLY dispersion and Kerr-Nonlinearity (split-step scheme)
    "SSFM_HONSE_symmetric":     Generalized Nonlinear Schrödinger Equation with simplified Raman-term TR (split-step scheme)
    "RK4IP_HONSE_symmetric":    Generalized Nonlinear Schrödinger Equation with simplified Raman-term TR (interaction picture split-step)
    "RK4IP_NSE_symmetric":      Nonlinear-Schrödinger-Equation ONLY dispersion and Kerr-Nonlinearity (interaction picture split-step)
    "RK4IP_RAMAN_symmetric":    Generalized Nonlinear Schrödinger Equation with  Raman-respose Function  (interaction picture split-step) 
    "RK4IP_RAMAN_fast":         Generalized Nonlinear Schrödinger Equation with  Raman-respose Function (interaction picture faster algorithm)

    The solver uses symmetric split step methods, such as Runge-Kutta methods.
    Depending on the model, characteristic Linear- and Nonlinear Operators are implemented.
    
    Absorbing boundaries only in symmetric algorithms
    """
    
    def __init__(self, window, material, model,AbsBoundary=False):
        """
        Input:
            - window (class):     Computational Window Parameters
            - material (class):   Material and pulse Parameters
            - model (String):     Name of Propagation model (see above)
            - AbsBoundary (bool): Specify, if absorbing boundaries are used
        """
        self.window    = window                              # Computational Window (Class)
        self.material  = material                            # Material and pulse Parameter (Class)
        self.A_t       = material.get_pulse(window)          # Initial pulse fitting into computational window (Array)
        self.LinOp     = self.LinearOperator()               # Linear Propagation Operator, frequency domain (Array)
        self.model     = model                               # Propagation Model (String)
        
        #DEFINE NONLINEAR OPERATOR DEPENDENT ON CHOSEN MODEL
        if (self.model == "SSFM_HONSE_symmetric" or self.model == "RK4IP_HONSE_symmetric"):
            self.NonLin    = self.NonLinearOperator_simple() # NonLinear Propagation Operator, time domain (Function(u))
        elif (self.model == "RK4IP_RAMAN_fast" or self.model == "RK4IP_RAMAN_symmetric"):  
            self.NonLin    = self.NonLinearOperator_Raman()  # NonLinear Propagation Operator, in frequency domain (Function(z,uw))
        elif (self.model =="SSFM_NSE_symmetric"):
            self.NonLin    = self.NonLinearOperator_Kerr()
        elif (self.model == "RK4IP_NSE_symmetric"):
             self.NonLin    = self.NonLinearOperator_Kerr_FD()
        else:
            print("Invalid model")
        
        if AbsBoundary:
            # -- ABSORBING BOUNDARIES IN TIME DOMAIN (ARRAY)
            self.Boundary_Cond = np.exp(-window.absorb_boundary_TD())
        else:
            # -- TRIVIAL CASE
            self.Boundary_Cond = np.exp(-np.zeros(len(window.t)))
        
        # DATA STRUCTURES THAT WILL ACCUMLATE RESULTS
        self.res_z=[0]; self.res_A=[self.A_t]


    def LinearOperator(self):
        """
        Define Linear Propagation Operator (in frequency domain)
        
        Input:
            - Computational_Window (class)
            - Material_Properties (class)
            
        Output:
            - LinOp: Linear Operator Array (const.) with len(window.w)
        
        Note: In this work the same for all used models.
        """
        
        w=self.window.w; fiber=self.material
        
        LinOp=np.zeros(len(w),dtype=complex)
        
        # -- ADD contributions from taylor-expansion of DISPERSION
        if len(fiber.beta)>2:
            coeffs=np.asarray([0,0]+[1j*(1j)**(n)*(-1j)**(n)/factorial(n) for n in range(2,len(fiber.beta))])
            coeffs=np.flip(coeffs*fiber.beta)
            dispersion=np.poly1d(coeffs)
            LinOp+=dispersion(w)
        
        # -- ADD contributions from frequency-dependent ABSORPTION PROFILE
        if fiber.alpha_params[0]!=0:
            a=fiber.alpha(w)
            LinOp-=a/2
        
        return LinOp   
    


    def NonLinearOperator_Kerr(self):
        """
        Define Nonlinear Propagation Operator (in time domain) for full-step propagation dz.
        Model: Only Kerr-Nonlinearity
        
        Source: Deiterding et al., JOURNAL OF LIGHTWAVE TECHNOLOGY, VOL. 31, NO. 12, JUNE 15, 2013 
        """
        
        fiber=self.material
        NonLin=lambda z, A_t: 1j*fiber.gamma*(np.abs(A_t)**2)
        return NonLin
    
    
    def NonLinearOperator_Kerr_FD(self):
        """
        Define Nonlinear Propagation Operator (in time domain) for full-step propagation dz.
        Model: Only Kerr-Nonlinearity
        
        Source: Deiterding et al., JOURNAL OF LIGHTWAVE TECHNOLOGY, VOL. 31, NO. 12, JUNE 15, 2013 
        """
        
        fiber=self.material
        _NR = lambda u: np.abs(u)**2*u
        return lambda z,uw: 1j*fiber.gamma*FT(_NR(IFT(uw)))
    
    
    def NonLinearOperator_simple(self):
        """
        Define Nonlinear Propagation Operator (in time domain) for full-step propagation dz.
        Model: Simplified Raman-term (TR)
        
        Source: Deiterding et al., JOURNAL OF LIGHTWAVE TECHNOLOGY, VOL. 31, NO. 12, JUNE 15, 2013 
        """
        
        w=self.window.w; dz=self.window.dz
        fiber=self.material
        NonLin=lambda z, A_t: 1j*fiber.gamma*(np.abs(A_t)**2 + [1j/fiber.omega0 - fiber.TR] * A_t
                                  * IFT(-1j*w*FT(np.conjugate(A_t))) +
                                         [2j/fiber.omega0 - fiber.TR]*np.conjugate(A_t)*IFT(-1j*w*FT(A_t)))
        return NonLin
     
    
    def NonLinearOperator_Raman(self):
        """
        Define Nonlinear Propagation Operator (in frequency domain) for full-step propagation dz.
        Model: Raman-response function
        
        Sources: O. Melchert, A. Demircan, SoftwareX 20 (2022) 101232
                 O. Melchert, A. Demircan, Computer Physics Communications 273 (2022) 108257
        """
        
        fiber = self.material
        fR    = fiber.fR;      tau1  = fiber.tau1;     tau2  = fiber.tau2
        w0    = fiber.omega0;  w     = self.window.w;  gamma = fiber.gamma
    
        hRw = ( tau1**2 + tau2**2 ) / ( tau1**2* ( 1 - 1j * w * tau2 )**2 + tau2**2 )
        # -- NONLINEAR FUNCTIONAL HANDLING THE RAMAN RESPONSE
        _NR = lambda u: ( 1 - fR ) * np.abs(u)**2 * u + fR * u * IFT(FT(np.abs(u)**2) * hRw)
        # -- APPLY FULL NONLINEAR OPERATION
        NonLin = lambda z, uw: 1j*gamma*(1.+w/w0)*FT(_NR(IFT(uw))) 
        
        return NonLin
    
    
    def SSFM_HONSE_symmetric(self):
        """
        Symmetric Split Step Method
        Local Error: O(dz^3)
        
        Returns:
            - Z-Grid of considered positions
            - Amplitudes A(z,t) (Time-Domain)
        """
        
        z=self.window.z; dz=self.window.dz
        
        # HALFSTEP for SYMMETRIC SSM and EXPONENTIAL for constant factor
        expLinOp=np.exp(self.LinOp*dz/2)
        
        for idx in range(1,z.size):
            
            self.A_t= IFT(expLinOp*FT(self.A_t))
            self.A_t = self.A_t* np.exp(self.NonLin(z,self.A_t)*dz)* self.Boundary_Cond
            self.A_t= IFT(expLinOp*FT(self.A_t))
            
            # -- PROGRESSBAR
            if (idx+1)%(self.window.z_N/10)==0:
                print((idx+1)/self.window.z_N*100,"%")
            
            # -- KEEP ONLY EVERY nSkip-TH FIELD CONFIGURATION 
            if idx%self.window.z_skip==0:
                self.res_z.append(z[idx])      # keep z-value
                self.res_A.append(self.A_t)    # keep field
    
        return np.asarray(self.res_z), np.asarray(self.res_A,dtype=complex)
    
    
    def RK4IP_RAMAN_symmetric(self):
        """
        Symmetric Split Step Method, RK4IP
        Local Error: O(dz^5)
        
        Returns:
            - Z-Grid of considered positions
            - Amplitudes A(z,t) (Time-Domain)
            
        Source: S. Balac, ESAIM: Math. Model. Num. Anal. 50 (2016)
        """
        z=self.window.z; dz=self.window.dz
        
         # -- EXPONENTIAL LINOP IS CONSTANT
        expLinOp=np.exp(self.LinOp*dz/2)
        expLinOp_m=np.exp(-self.LinOp*dz/2)
        
        
        #SOLVE IN FOURIER DOMAIN
        A_w  = FT(self.A_t)
        
        for idx in range(1,z.size):
            A_w  = expLinOp*A_w
            # -- RK4  FULL-STEP FOR NONLIN
            A_w  = RK4IP(self.NonLin,expLinOp,expLinOp_m, 0, A_w, dz)
            A_w  = expLinOp*A_w
            
            # -- PROGRESSBAR
            if idx%(self.window.z_N/10)==0:
                print(idx/self.window.z_N*100,"%")
            
            
            # -- KEEP ONLY EVERY nSkip-TH FIELD CONFIGURATION 
            if idx%self.window.z_skip==0:
                A_t  = IFT(A_w) #TRANSFORM TO TIME DOMAIN
                self.res_z.append(z[idx]) # keep z-value
                self.res_A.append(A_t)    # keep field
    
        return np.asarray(self.res_z), np.asarray(self.res_A,dtype=complex)
    
    def RK4IP_RAMAN_fast(self):
        """
        Symmetric Split Step Method: saves computational evaluations (see source)
        Local Error: O(dz^5)
        
        Returns:
            - Z-Grid of considered positions
            - Amplitudes A(z,t) (Time-Domain)
        Source:
            - S. Balac, ESAIM: Math. Model. Num. Anal. 50 (2016)
        """
        z=self.window.z; dz=self.window.dz
        # -- EXPONENTIAL LINOP IS CONSTANT
        expLinOp=np.exp(self.LinOp*dz/2)
        
        # -- SOLVE IN FOURIER DOMAIN
        A_w  = FT(self.A_t)
        
         
        for idx in range(1,z.size):
            
            Aw_I = expLinOp*A_w
            # -- COMPUTE ALL FOUR STAGES OF THE RK SCHEME
            k1 = expLinOp*self.NonLin(z, A_w)*dz
            k2 = self.NonLin(z+dz*0.5, Aw_I + k1/2)*dz
            k3 = self.NonLin(z+dz*0.5, Aw_I + k2/2)*dz
            k4 = self.NonLin(z+dz,expLinOp*Aw_I + k3)*dz
            # -- PERFORM FIELD UPDATE
            A_w = expLinOp*(Aw_I + k1/6 + k2/3 + k3/3) + k4/6
        
        
            # -- PROGRESSBAR
            if idx%(self.window.z_N/10)==0:
                print(idx/self.window.z_N*100,"%")
            
            # -- KEEP ONLY EVERY nSkip-TH FIELD CONFIGURATION 
            if idx%self.window.z_skip==0:
                A_t  = IFT(A_w) # TRANSFORM TO TIME DOMAIN
                self.res_z.append(z[idx]) # keep z-value
                self.res_A.append(A_t)    # keep field
    
        return np.asarray(self.res_z), np.asarray(self.res_A,dtype=complex)
    
   
    def compute_model(self):
        """
        match algorithm to chosen mode
        """
        print("model:     ",self.model)
        if self.model =="SSFM_HONSE_symmetric":
            return self.SSFM_HONSE_symmetric()
        elif self.model =="RK4IP_RAMAN_symmetric":
            return self.RK4IP_RAMAN_symmetric()  
        elif self.model =="RK4IP_RAMAN_fast":
            return self.RK4IP_RAMAN_fast()
        elif self.model =="SSFM_NSE_symmetric":
            return self.SSFM_HONSE_symmetric()
        elif self.model == "RK4IP_NSE_symmetric":
            return self.RK4IP_RAMAN_symmetric()  
        elif self.model == "RK4IP_HONSE_symmetric":
            return self.RK4IP_RAMAN_symmetric()
        

       
def RK4IP(fun, lin_p, lin_m, z, uw, dz):
    """
    Fourth-order Runge-Kutta Interaction Picture
    Local Error: O(dz^5)
    
    Sources:
        - C. Runge, Math. Ann. 46 (1895)
        - W. Kutta,  Z. Math. Phys. 46 (1901)
        - S. Balac, ESAIM: Math. Model. Num. Anal. 50 (2016)
    """
    k1 = lin_p*fun(z, lin_m*uw)
    k2 = fun(z, uw + dz * 0.5 * k1)
    k3 = fun(z, uw + dz * 0.5 * k2)
    k4 = lin_m*fun(z, lin_p*(uw + dz * k3))
    return uw + dz * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0  