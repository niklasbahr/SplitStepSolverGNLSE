#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NIKLAS BAHR, 26.09.2023

MASTERARBEIT - Zeno-Effect in Nonlinear Optics

THEORETICAL OPTICS AND COMPUTATIONAL PHOTONICS

CONTENT:
    - VirtualLab (see Ch. 5.6 in thesis)
"""

# -- OWN MODULES
from src.Material_Window import Material_Properties, Computational_Window
from src.SplitStepSolver import SplitStepSolver
from src.export_import import save_params, import_params
from src.figures import figure_2a


def VirtualLab():
    # INITIALIZE: Computational window
    window = Computational_Window(t_max=5, t_N=2**12, z_max=.16, z_N=10000, z_skip=50)
    
    # INITIALIZE: Fiber and Pulse
    material=Material_Properties()
    
    # DEFINE Fiber Properties
    material.define_fiber(alpha=[0,0,0,"flat"], beta=[0,0,-0.011,0.08e-3], gamma=0.15)
    
    # DEFINE Pulse Properties
    material.define_pulse(t0=0.03, P0=1000, omega0=2.26e3, shape='soliton-noise')
    
    # INITIALIZE: SOLVER - window, material, model, absorbing boundaries
    solver = SplitStepSolver(window,material,'RK4IP_RAMAN_symmetric',False)
    
    # COMPUTE SOLUTION
    z, Azt = solver.compute_model()
    
    # SAVE RESULTS
    save_params(material, window, Azt, oName= 'example.zip', path='res/')

    # CREATE FIGURE
    figure_2a(z/material.soliton_period(), window.t, Azt, tLim=(-0.5,1.5), wLim=(-750,750), path='fig/', oName='example')
    

VirtualLab()