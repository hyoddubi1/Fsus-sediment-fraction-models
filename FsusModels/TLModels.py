# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 16:47:35 2022

@author: user
"""


import pandas as pd
import numpy as np
#import hsdatapkg
import FsusModels


def stattable(y,ypred,model = 'model'):
#    ys2diff = ypred.subtract( y,axis = 0)
    
    ys2diff = ypred-y

    pbias = ys2diff.sum() / y.sum() * 100
    
    mse = (ys2diff**2).mean()
    
    r2 = 1 - ( ys2diff**2).sum() / ((y - y.mean())**2).sum()
   

    StatTable = pd.DataFrame(data = [mse,pbias,r2], index=['MSE','PBIAS','R2'])
    StatTable = StatTable.rename(columns = {0:model})
    
    return StatTable

def predict(nv,model):
    
    Fsus = FsusModels.main(nv,model = model)

    
    return Fsus

def recalytemp(nv):
    
    if nv == 3:
        Dataset = pd.read_csv("Dataset/Input_3vars.csv")
    
    if nv == 5:
        Dataset = pd.read_csv("Dataset/Input_5vars.csv")
    
    if nv == 6:
        Dataset = pd.read_csv("Dataset/Input_6vars.csv")

    y = Dataset["$F_{sus}$"].values
    
    return y


    
def Ctoqt(C,q):
    
#    gamma = 2.65
    #kg/s
    
    qt = C * 1000 * q
    
    return qt

def CtoQt(C,Q,rhow):
    
   #kg/s
    Qt = C * Q*1000# = 1000 kg/s = cms
    
    return Qt



def EH1967(u,Rh,d50,S0):
    Gs = 2.65 
    g = 9.81
    Ct = 0.05 * (Gs / (Gs - 1)) * u * S0 / np.sqrt((Gs - 1) * g * d50) * Rh * S0 / (d50 * (Gs-1))
    #Cw
    return Ct
    
def AW1973(u,h,d50,S0,nu):
#    nu = 10 **(-6)
    g = 9.81
    Gs = 2.65
    dstar = d50 * (g * (Gs-1) / nu**2)**(1/3)
    ustar = np.sqrt(g * h * S0)
    
    if dstar > 60:
        caw1 = 0
        caw2 = 0.025
        caw3 = 0.17
        caw4 = 1.5
    else:
        caw1 = 1 - 0.56 * np.log10(dstar)
        caw2 = np.exp(2.86 * np.log10(dstar) - (np.log10(dstar))**2 - 3.53)
        caw3 = 0.23 / np.sqrt(dstar) +0.14
        caw4 = 9.66 / dstar + 1.34
        
    caw5 = ustar ** caw1 / np.sqrt((Gs-1) * g * d50) *( u / (np.sqrt(32) * np.log10(10 * h / d50)))**(1-caw1)
    
    Ct = caw2 * Gs * (d50 / h) * (u / ustar) ** caw1 *(caw5 / caw3 - 1) ** caw4
    #Cw
    return Ct

def SH1972(u,w,S0):

    Sh = (u * S0**0.57159 / w **0.31988) ** 0.00750189
    logC = [-107404.459+324214.747 * Sh - 326309.589 * Sh**2 + 109503.872 * Sh **3]
    
    Ct = 10 ** (logC)
    #Cppm
    return Ct

def Y1979(u,h,d50,S0,nu):
#    nu = 10 **(-6)
    g = 9.81
    Gs = 2.65
    dstar = d50 * (g * (Gs-1) / nu**2)**(1/3)
    ws = 8 * nu / d50 * (np.sqrt(1 + 0.0139 * dstar ** 3)-1)
    ustar = np.sqrt(g * h * S0)
    
#    roughness = ustar * d50 / nu
    
#    if roughness < 70:
#        Vcw = 2.5 / (np.log10(roughness) - 0.06) + 0.66
#    else:
#        Vcw = 2.05
##    
    logC = 5.165 - 0.153 * np.log10(ws * d50 / nu) - 0.297 * np.log10(ustar / ws) + (1.780
                                   - 0.360 * np.log10(ws * d50 / nu) 
                                   - 0.480 * np.log10(ustar / ws)) * np.log10(u * S0 / ws)
#        
#    if d50 <2:        #sand
#        logC = 5.435 - 0.286 * np.log10(ws * d50 / nu) - 0.457 * np.log10(ustar / ws) + (1.799
#                               - 0.409 * np.log10(ws * d50 / nu) 
#                               - 0.314 * np.log10(ustar / ws)) * np.log10(u * S0 / ws - Vcw * S0)
#    else: # gravel
#        logC = 6.681 - 0.633 * np.log10(ws * d50 / nu) - 4.816 * np.log10(ustar / ws) + (2.784
#                               - 0.305 * np.log10(ws * d50 / nu) 
#                               - 0.282 * np.log10(ustar / ws)) * np.log10(u * S0 / ws - Vcw * S0)
        
    
    Ct = 10 ** (logC)
    #Cppm
    return Ct
def Y19792(u,h,d50,S0,nu):
#    nu = 10 **(-6)
    g = 9.81
    Gs = 2.65
    dstar = d50 * (g * (Gs-1) / nu**2)**(1/3)
    ws = 8 * nu / d50 * (np.sqrt(1 + 0.0139 * dstar ** 3)-1)
    ustar = np.sqrt(g * h * S0)
    
    roughness = ustar * d50 / nu
    
    if roughness < 70:
        Vcw = 2.5 / (np.log10(roughness) - 0.06) + 0.66
    else:
        Vcw = 2.05
#    
#    logC = 5.165 - 0.153 * np.log10(ws * d50 / nu) - 0.297 * np.log10(ustar / ws) + (1.780
#                                   - 0.360 * np.log10(ws * d50 / nu) 
#                                   - 0.480 * np.log10(ustar / ws)) * np.log10(u * S0 / ws)
        
    if d50 <2:        #sand
        logC = 5.435 - 0.286 * np.log10(ws * d50 / nu) - 0.457 * np.log10(ustar / ws) + (1.799
                               - 0.409 * np.log10(ws * d50 / nu) 
                               - 0.314 * np.log10(ustar / ws)) * np.log10(u * S0 / ws - Vcw * S0)
    else: # gravel
        logC = 6.681 - 0.633 * np.log10(ws * d50 / nu) - 4.816 * np.log10(ustar / ws) + (2.784
                               - 0.305 * np.log10(ws * d50 / nu) 
                               - 0.282 * np.log10(ustar / ws)) * np.log10(u * S0 / ws - Vcw * S0)
        
    
    Ct = 10 ** (logC)
    #Cppm
    return Ct

def O2016(u,h,d50,S0,nu):
#    nu = 10 **(-6)
    g = 9.81
    Gs = 2.65

    ustar = np.sqrt(g * h * S0)
    
    P = u / np.sqrt((Gs-1) *g * d50)
    J = np.exp((np.log(S0))**3)
    L = np.exp((np.log(h/d50))**2)
    R = ustar * d50 / nu

    Ct = 34.45 * ( P**3.239 * J**0.005) / (L**0.066 * R**0.146)
    #Cppm
    return Ct

def MW2001(u,h,d50,S0,nu):
#    nu = 10 **(-6)
    g = 9.81
    Gs = 2.65
    dstar = d50 * (g * (Gs-1) / nu**2)**(1/3)
    ws = 8 * nu / d50 * (np.sqrt(1 + 0.0139 * dstar ** 3)-1)
    
    denom = (Gs-1) * g * h * ws * (np.log(h/d50))**2
    
    Psi = u**3 / (denom)
    
    Ct = 1430 * (0.86 + np.sqrt(Psi)) * Psi**(1.5) / (0.016 + Psi)
    #Cppm
    return Ct


def ModelColumn2Predict(df):
    models = df.columns[1:]
    
    model = models[0]
    df2 = df[['Observed',model]]
    
    df2['Model'] = model
    df2 = df2.rename(columns = {model:'Predicted'})
    
    for model in models[1:]:
        tempdf = df[['Observed',model]]
        tempdf['Model'] = model
        tempdf = tempdf.rename(columns = {model:'Predicted'})
        df2 = pd.concat([df2,tempdf],axis = 0)

    
    
    return df2


def EvalPubModels(Input):
    Q = Input.Q
    u = Input.U
    h = Input.H
    d50 = Input['$d_50$']
    S0 = Input['$S_0$']
    T = Input['Temp.']
    nu = Input['nu']
    rhow = FsusModels.DensWaterCivan(T)
    rhow[np.isnan(rhow)] = 1000
    
    
    
    modelStatTable = stattable(np.array(1),np.array(1),model ="dummy")
    
    TLobs = Input['TL']
    
    TLpred = Input[['TL']].rename(columns={'TL':'Observed'})
    
                #=========================================================
    tempmodel = 'EH1967'
    TL = EH1967(u,h,d50,S0)
    TL = CtoQt(TL,Q,rhow)
    
    
    TLnotnan = ~np.isnan(TL)
#    print("\n\nlen\n\n",np.sum(TLnotnan))
    TLeval = TL[TLnotnan]
    modelStatTable = pd.concat([modelStatTable, stattable(TLobs[TLnotnan],TLeval,model=tempmodel)],axis = 1)
    TLpred[tempmodel] = TL
            #=========================================================
    tempmodel = 'AW1973'
    
    TL = np.zeros(len(TL))
    for i in range(len(TL)):
        TL[i] = AW1973(u.values[i],h.values[i],d50.values[i],S0.values[i],nu.values[i])
    TL = CtoQt(TL,Q,rhow)
    
    TLnotnan = ~np.isnan(TL)
#    print("\n\nlen\n\n",np.sum(TLnotnan))
    TLeval = TL[TLnotnan]
    modelStatTable = pd.concat([modelStatTable, stattable(TLobs[TLnotnan],TLeval,model=tempmodel)],axis = 1)
    TLpred[tempmodel] = TL
        #=========================================================
    tempmodel = 'Y1979'
    TL = Y1979(u,h,d50,S0,nu) 
    
    TL = CtoQt(TL,Q,rhow)/1e6
    
    TLnotnan = ~np.isnan(TL)
#    print("\n\nlen\n\n",np.sum(TLnotnan))
    TLeval = TL[TLnotnan]
    modelStatTable = pd.concat([modelStatTable, stattable(TLobs[TLnotnan],TLeval,model=tempmodel)],axis = 1)
    TLpred[tempmodel] = TL
    
    #=========================================================
    tempmodel = 'MW2001'
    TL = MW2001(u,h,d50,S0,nu)
    TL = CtoQt(TL,Q,rhow)/1e6
    TLnotnan = ~np.isnan(TL)
#    print("\n\nlen\n\n",np.sum(TLnotnan))
    TLeval = TL[TLnotnan]
    modelStatTable = pd.concat([modelStatTable, stattable(TLobs[TLnotnan],TLeval,model=tempmodel)],axis = 1)
    TLpred[tempmodel] = TL
    
    #=========================================================
    tempmodel = 'O2016'
    TL =  O2016(u,h,d50,S0,nu)
    TL = CtoQt(TL,Q,rhow)/1e6
    TLnotnan = ~np.isnan(TL)
#    print("\n\nlen\n\n",np.sum(TLnotnan))
    TLeval = TL[TLnotnan]
    modelStatTable = pd.concat([modelStatTable, stattable(TLobs[TLnotnan],TLeval,model=tempmodel)],axis = 1)
    TLpred[tempmodel] = TL
    

    
#    legendEH =  "E-H (1967): $R^2$ = " + str(np.round(R2EH,2))
#    legendAW =  "A-W (1973): $R^2$ = " + str(np.round(R2AW,2))
#    legendY =  "Yang (1979): $R^2$ = " + str(np.round(R2Y,2))
#    legendO =  "Okcu et al. (2016): $R^2$ = " + str(np.round(R2O,2))
#    legendMW =  "M-W (2001): $R^2$ = " + str(np.round(R2MW,2))
    
    
    return TLpred , modelStatTable
