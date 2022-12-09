# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 09:53:22 2022

@author: user
"""

import numpy as np
from scipy import integrate
import FsusModels


def J1sp(zstar, Ro):
    # Einstein integral J1's prime -> J1 = int(J1sp)
    f = ((1 - zstar)/zstar )**Ro
    
    return f

def J2sp(zstar, Ro):
    # Einstein integral J1's prime -> J1 = int(J1sp)
    
    f = np.log(zstar) *  ((1 - zstar)/zstar )**Ro
    
    return f
    

def fall_vel(ds, nu = 10**-6):
    G = 1.65
    g = 9.81
    
    
    dstar = ds * ((G-1)*g/ nu**2 )**(1/3)
    
    w = 8 * nu * ds * ((1 + 0.0139 * dstar**3)**0.5 -1)
    
    return w

def SEMEPFsus(dss,ds,h,ustar, nu = 10**-6, T = None):
    
    if T:
        nu = FsusModels.KinVis_Cels(T)
    
    beta = 1 # diffusivity
    kappa = 0.4 # von Karman coefficient
    ws =  fall_vel(dss, nu = nu) # settling velocity
    
    Ro = ws / (beta * kappa * ustar)
    B =  2 * ds / h
    
    # Einstein integral in numerical integration
    
    J1s = integrate.quad(J1sp,B,1,args=(Ro))
    J2s = integrate.quad(J2sp,B,1,args=(Ro))

    J1s = J1s[0]    
    J2s = J2s[0]
    
    # calcuation
    numerator = 0.216 * (B**(Ro-1) / (1 - B)**Ro) * (np.log(30 * h / ds) * J1s + J2s)
    
    denominator = 1 + numerator
    
    Fsus = numerator / denominator
    
    return Fsus



if __name__ == "__main__":
    
    g = 9.81
    
    d65 = 4.852 # d65 of bed material
    d50ss = 15.121 # d50 of suspended sediment
  
    hs=[1.24,1.05,0.7,1.38,1.35,1.32,1.39,1.31,0.5,0.48]
    h1 = 34.72
    h2 = 27.72
    L = 5800
    h = np.mean(np.array(hs))
    
    #청미
    
#    d65 = 0.846 # d65 of bed material
#    d50ss = 13.866 # d50 of suspended sediment
#    h1 = 51.94
#    h2 = 42.28
#    L= 10900
#    hs=1.5
    
    
#    S = 0.0001
    S = - (h2 - h1)/L

    ustar = np.sqrt(g * h * S)
  
    
    ds = d65 * 0.001
    dss = d50ss * 10**-6
    
    
    Fsus = SEMEPFsus(dss,ds,h,ustar)
    
    print("Fsus=",Fsus)