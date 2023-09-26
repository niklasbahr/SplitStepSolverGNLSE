#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NIKLAS BAHR, 11.11.2022

MASTERARBEIT - Zeno-Effect in Nonlinear Optics

THEORETICAL OPTICS AND COMPUTATIONAL PHOTONICS

CONTENT:
    - save_last
    - save_params
    - import_params
"""

# -- USED LIBRARIES
import os
import json
import inspect
import numpy as np
from zipfile import ZipFile

# -- OWN MODULES
from src.Material_Window import Material_Properties, Computational_Window



def save_last(material, window, Azt, oName=None, path='/Users/niklas.bahr/Desktop/Masterarbeit/01_Parameter/'):
    """
    saves:
        - complex field amplitude at position z_max (last value of Azt) -> Numpy Array
        - material (object of class Material_Properties) -> JSON
        - window (object of class Computational_Window) ->JSON
    
    input: filename and path
    """
    
    attributes = inspect.getmembers(material, lambda a:not(inspect.isroutine(a)))
    material_params=dict([a for a in attributes if not(a[0].startswith('__') and a[0].endswith('__'))])
    
    attributes = inspect.getmembers(window, lambda a:not(inspect.isroutine(a)))
    window_params=dict([a for a in attributes if not(a[0].startswith('__') and a[0].endswith('__'))])
    
    t = window.t
    z = window.z
    
    del window_params["t"]
    del window_params["z"]
    del window_params["w"]
    
    
    with open("material.json", "w") as fp:
        json.dump(material_params,fp, indent = 0)
    with open("window.json", "w") as fp:
        json.dump(window_params,fp, indent = 0)  
    np.savez("results.npz",z=z,t=t,Azt=Azt[-1])#only save field at last propagation-step
    
    if oName is None:
        filename="last_z"+str(window_params["z_max"])+"_seed"+str(material_params["noise_seed"])+".zip"
    else:
        filename=oName
                
    path+=filename
  
    try:
        zipObj = ZipFile(path, 'w')

        zipObj.write('material.json')
        zipObj.write('results.npz')
        zipObj.write('window.json')
        zipObj.close()
        os.remove("material.json")
        os.remove("window.json")
        os.remove("results.npz")
        return material_params, window_params
    except:
        print("Error: Path not found. Adapt code above")
        return False
    


def save_params(material, window, Azt, oName=None, path='/Users/niklas.bahr/Desktop/Masterarbeit/01_Parameter/'):
    """
    This function saves properties of given Material (Fiber/Pulse) and Computational Window in ZIP-File.
    Use string variables "oName" and "path" to define, where the ZIP-files shall be saved!
    
    First inspect all atributes in class and keep their values in a dictionary.
    Delete Numpy-Arrays t,z from window-dictionary (not possible to save arrays in JSON)
    Save each dictionary via json.
    Save Both Json-files in a dictionary and add them in a ZIP-file
    Delete single Json-files and only keep ZIP-file
    
    input:
     - material: Object of class Material_Properties
     - window: Object of class Computational_Window
     - oName: needs to end with ".zip". If not given, an automatic name is given based on parameters. May be not unique!!!
    
    output:
     - material_params, window_params: properties as dictionaries   
    """
    
    attributes = inspect.getmembers(material, lambda a:not(inspect.isroutine(a)))
    material_params=dict([a for a in attributes if not(a[0].startswith('__') and a[0].endswith('__'))])
    
    attributes = inspect.getmembers(window, lambda a:not(inspect.isroutine(a)))
    window_params=dict([a for a in attributes if not(a[0].startswith('__') and a[0].endswith('__'))])
    
    t = window.t
    z = window.z
    w = window.w
    
    del window_params["t"]
    del window_params["z"]
    del window_params["w"]
    
    
    with open("material.json", "w") as fp:
        json.dump(material_params,fp, indent = 0)
    with open("window.json", "w") as fp:
        json.dump(window_params,fp, indent = 0)  
    np.savez("results.npz",z=z,t=t,Azt=Azt,w=w)
    
    
    if oName is None:
        filename=str(material_params["shape"])\
                    +"_T"+str(material_params["t0"])\
                      +"_P"+str(material_params["P0"])\
                        +"_a"+str(material_params["alpha_params"][0])\
                          +"_2b"+str(material_params["beta"][2])\
                            +"_3b"+str(round(material_params["beta"][3],6))\
                              +"_c"+str(material_params["gamma"])\
                    +"_z"+str(window_params["z_max"])\
                      +"_Nz"+str(window_params["z_N"])\
                        +"_t"+str(window_params["t_max"])\
                          +"_Nt"+str(window_params["t_N"])\
                    +".zip"
    else:
        filename=oName
                
    path+=filename
  
    try:
        zipObj = ZipFile(path, 'w')

        zipObj.write('material.json')
        zipObj.write('results.npz')
        zipObj.write('window.json')
        zipObj.close()
        os.remove("material.json")
        os.remove("window.json")
        os.remove("results.npz")
        return material_params, window_params
    except:
        print("Error: Path not found. Adapt code above")
        return False



def import_params(filename='sample.zip',path='/Users/niklas.bahr/Desktop/Masterarbeit/01_Parameter/'):
    """
    This function loads properties of Material (Fiber/pulse) and Computational Window stored in a ZIP-File.
    Use string variable "path" to define, where the ZIP-files shall be searched!
    
    First read ZIP-File and open both dictionaries.
    Create objects with corresponding values. If Input-Parameters of classes are changed, also change here!
    
    input:
     - fileame: string with name of ZIP-File (needs to end with .zip)
    
    output:
     - material, window: Obbjects of classes Material_Properties() and Computational_Window()
    """
    
    path+=filename
    
    archive = ZipFile(path, 'r')
    with archive.open("material.json","r") as fp:#r - open file in read mode
       data = json.load(fp)
       material=Material_Properties()
       if len(data["alpha_params"])==3.:
           data["alpha_params"].append("erf")
       material.define_fiber(alpha=data["alpha_params"],beta=data["beta"],gamma=data["gamma"],TR=data["TR"])
       material.define_pulse(P0=data["P0"],t0=data["t0"],shape=data["shape"],order=data["order"],omega0=data["omega0"]) 
    
    with archive.open("window.json","r") as fp:#r - open file in read mode
       data = json.load(fp)
       window=Computational_Window(t_max=data["t_max"],t_N=data["t_N"],z_max=data["z_max"],z_N=data["z_N"],z_skip=data["z_skip"])   
    try: 
        with archive.open("results.npz","r") as fp:#r - open file in read mode
           data = np.load(fp)
           t=data["t"]
           z=data["z"]
           Azt=data["Azt"]
    except:
        t=[]
        z=[]
        Azt=[]
       
    return material, window, z, t, Azt 