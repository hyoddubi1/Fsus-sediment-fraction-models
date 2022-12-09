# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 11:33:14 2022

@author: Hyoseob Noh


"""
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import joblib

def DynVis(T):
    #Vogel's equation #mPa
    A = -3.7188
    B = 578.919
    C = -137.546
    
    po = A + B / (C + T)
    
    return np.exp(po)

def DensWaterCivan(T):
    #Civan2007
    A =1.2538
    B = -1.4496 * 10**3
    C = 1.2971 * 10**5
    T2 = T + 175      
    
    Reldev = A + B / T2 + C / T2**2
    
    
    rho = (1 - np.exp(Reldev)) * 1065
    
    return rho

def KinVis_Cels(TCels):
    #NIST
    T0 = 273.15

    TKelv = T0 + TCels
    mu = DynVis(TKelv)
#    rho= DensWater(T)
#    rho= DensWaterbeta(T)
    rho = DensWaterCivan(TCels)
    
    mukPa = mu /10 **3
    mukgm2 = mukPa 
    nu = mukgm2/rho
    
    return nu

def MakeDimLess(U,h,W = None , d50=None,S0=None,nu=10**-6, T=None, ustar = None, rhos = None):
    # Generating dataframe of variables that are used in the empirical model by Noh et al.
    # Please match the data size
    # Basic dimensions
    # d50: mm
    # T: Celsius degree -> For nu recalcuation
    # else: m or m/s
    g = 9.81 # m/s2    
    beta = 1 #Rouse number coefficient
    kappa = 0.41 # von Karman coefficient
    
    U = np.array(U)
    h = np.array(h)
    W = np.array(W)
    d50 = np.array(d50)
    S0 = np.array(S0)
    nu = np.array(nu)
    T = np.array(T)
    ustar = np.array(ustar)
    rhos = np.array(rhos)
    
    nanT = np.isnan(T)
    T[nanT] = 20
    
    #=================================================
    DimLessdf = pd.DataFrame()
    dflen = np.max([np.size(U),np.size(h),np.size(W),np.size(d50),np.size(S0),np.size(nu),
                    np.size(T),np.size(ustar),np.size(rhos)])
    DimLessdf["dummy"] = np.arange(0,dflen)
    #=================================================
    # Material properties
    

    if np.max(T == None):
        DimLessdf["nu"] = nu
        DimLessdf["rhow"] = 1000
    else:
        DimLessdf["nu"] =  KinVis_Cels(T)
        DimLessdf["rhow"] = DensWaterCivan(T)

        
    if np.max(rhos == None):
        DimLessdf["rhos"] = 2650
    else:
        DimLessdf["rhos"] = rhos
        
    DimLessdf["Gs"] = DimLessdf["rhos"] / DimLessdf["rhow"]
    #=================================================
 
    if np.max(S0 != None):
        DimLessdf["S0"] = S0
        
    else:
        DimLessdf["S0"] = np.nan

    if np.max(ustar == None):
        if S0 != None:
            DimLessdf["ustar"] = np.sqrt(g*h*S0)
        else:
            DimLessdf["ustar"] = np.nan
    else:
        DimLessdf["ustar"] = ustar   
        
    if np.max(W != None):
        DimLessdf["W/h"] = W/h
    else:
        DimLessdf["W/h"] = np.nan
    #=================================================
    # Flow related Dimensionless numbers
        
        
    DimLessdf["Fr"] = U/np.sqrt(g*h)
    DimLessdf["Reh"] = U*h/DimLessdf.nu
    DimLessdf["Restar"] = DimLessdf.ustar * h /DimLessdf.nu

    
    if np.max(d50 != None):
        DimLessdf["d50"] = d50 /1000
        DimLessdf["dstar"] = DimLessdf["d50"] * (g *( DimLessdf["Gs"] - 1) / DimLessdf["nu"]**2)**(1/3)
        DimLessdf["ws"] = 8 * DimLessdf["nu"] / DimLessdf["d50"] * ( (1 + 0.0139 * DimLessdf["dstar"]**3)**0.5- 1)
        
        DimLessdf["Frd"] = U/np.sqrt(g* (DimLessdf.Gs-1) *DimLessdf.d50)
        DimLessdf["Ro"] = DimLessdf.ws / DimLessdf.ustar /(beta * kappa)
        
        DimLessdf["US0/ws"] = U * DimLessdf.S0 / DimLessdf.ws
        DimLessdf["Rew"] = DimLessdf.d50 * DimLessdf.ws / DimLessdf.nu
        
    DimLessdf = DimLessdf.drop(columns=["dummy"])
    
    return  DimLessdf

def dftoX(DimLessdf,n_var = 3):
    # Reshaping DimLessdf into the input structure of the empirical equations
    
    if n_var == 3:
        # The variables are as below
        # X1 = Re_h
        # X2 = Fr
        # X3 = Fr_d
        X = DimLessdf[["Reh","Fr","Frd"]].values
    elif n_var == 5:
        # The variables are as below
        # X1 = W/h
        # X2 = d_*
        # X3 = Re_h
        # X4 = Fr_d
        # X5 = Re_w
        X = DimLessdf[["W/h","dstar","Reh","Frd","Rew"]].values
    elif n_var == 6:
        # The variables are as below
        # X1 = W/h
        # X2 = US0/ws
        # X3 = Re_h
        # X4 = Re_*
        # X5 = Fr_d
        # X6 = Ro
        X = DimLessdf[["W/h","US0/ws","Reh","Restar","Frd","Ro"]].values
    
    return X


def CheckDimension(X):
    if np.ndim(X) == 1:
        X = np.reshape(X,(1,-1))
        print("Input data is reshaped to 2 dimensional array (1,n_varibles).")
    
    return X

def SVR3(X):
    # Fsus equation that makes 
        # X1 = Re_h
        # X2 = Fr
        # X3 = Fr_d

    X = CheckDimension(X)
    
    gridrfeFsus = joblib.load("./ModelFiles/SVRmodels/SVR3.pkl")
    
    
    return gridrfeFsus.predict(X)

def SVR5(X):
    # Fsus equation that makes 
        # X1 = W/h
        # X2 = d_*
        # X3 = Re_h
        # X4 = Fr_d
        # X5 = Re_w

    X = CheckDimension(X)
    
    gridrfeFsus = joblib.load("./ModelFiles/SVRmodels/SVR5.pkl")
    
    
    return gridrfeFsus.predict(X)
    


def MGGP3(X):
    # Fsus equation that makes 
        # X1 = Re_h
        # X2 = Fr
        # X3 = Fr_d

    sc_X = joblib.load("./ModelFiles/Scalers/sc_X3.pkl")
    X = CheckDimension(X)
    X = sc_X.transform(X)
    ReH = X[:,0]
    Fr = X[:,1]
    Fr_d = X[:,2]
    
    Fsus = 0.406* np.exp(np.exp(- 6.0* Fr_d - 3.0* ReH) - Fr**2 *ReH**3) - 1.97 *np.exp(- ReH) - 0.779* np.exp(Fr_d**2) + 0.779 *np.exp(- ReH**3) + 1.45* Fr_d**2 + 1.77

    return Fsus


def MGGP5(X):
    # Fsus equation that makes 
        # X1 = W/h
        # X2 = d_*
        # X3 = Re_h
        # X4 = Fr_d
        # X5 = Re_w

    sc_X = joblib.load("./ModelFiles/Scalers/sc_X5.pkl")
    X = CheckDimension(X)
    X = sc_X.transform(X)
    Wh = X[:,0]
    dstar = X[:,1]
    Reh = X[:,2]
#    Frd = X[:,3]
    Rew = X[:,4]
    
#    sin = np.sin
#    cos = np.cos
    tanh = np.tanh
    exp = np.exp
#    sqrt = np.sqrt
    
    Fsus = 0.365 *exp(exp(-(tanh(Reh))/(Reh + dstar))/tanh(exp(- Rew)**(Reh *dstar))) - 0.549* dstar - 0.0521 *exp(3.0 *exp(- Reh)) - 0.0521* Reh + 0.222 *Wh *dstar - 0.0521* (Wh**dstar)**(1/2) + 0.708
    #0.316143
    return Fsus
#==========================================================


def Operon3(X):
    # Fsus equation that makes 
        # X1 = Re_h
        # X2 = Fr
        # X3 = Fr_d
    
    sc_X = joblib.load("./ModelFiles/Scalers/sc_X3.pkl")
    X = CheckDimension(X)
    X = sc_X.transform(X)
    X1 = X[:,0]
    X2 = X[:,1]
    X3 = X[:,2]

    Fsus = (1.012*(2.616*X1 - 11.552*X2 + (20.192*X2 - 1.331)/((7.505*X1 - 0.567*X2 + (45.229*X3)/((11.916304*X2**2)/(387.893025*X1**2 + 1.0) + 1.0)**(1/2) - 0.04)**2 + 1.0)**(1/2) - (1.0*(3.364*X2 - 1.587))/(8330.395441*X1**2 + 1.0)**(1/2) + (3421.821*X3 + 0.005)*(0.075*X1 + 0.004*X2 + 0.005)))/((0.711*X1 - 11.392*X2 + (0.057*X1 + 0.015)*(9.269*X1 + 3739.117*X3 + 31.422))**2 + 1.0)**(1/2) - 0.009
 
    return Fsus
    
def Operon5(X):
        # X1 = W/h
        # X2 = d_*
        # X3 = Re_h
        # X4 = Fr_d
        # X5 = Re_w

    sc_X = joblib.load("./ModelFiles/Scalers/sc_X5.pkl")
    X = CheckDimension(X)
    X = sc_X.transform(X)
        
    X1 = X[:,0]
    X2 = X[:,1]
    X3 = X[:,2]
    X4 = X[:,3]
    X5 = X[:,4]
    
    Fsus = 0.499*X1 - (27.784*X3 - 0.657*X2 - 2.446*X4 + 0.563/(38808.212*X5**2 + 1)**(1/2) + 1.331)/(288.388*X3**2 + 1)**(1/2) - (0.2*(2878.0*X1 + 1345.0*X2 + 2235.0*X4))/200/(5670.843*X3**2 + 1)**(1/2) + 2.622
#    Fsus = ((-0.000001) + (1.000002 * ((0.498866 * X1) + (2.622187 + (((((((-27.784449) * X3) + ((-0.562650) / (sqrt(1 + ((-196.997635) * X5) ** 2)))) + ((-1.330822) + (2.445854 * X4))) + (0.657379 * X2)) / (sqrt(1 + ((-16.982428) * X3) ** 2))) + (((((-2.234883) * X4) + ((-1.344779) * X2)) + ((-2.877887) * X1)) / (sqrt(1 + ((-75.304794) * X3) ** 2))))))))
    return Fsus


def GPGOMEA3(X):
    # Fsus equation that makes 
        # X1 = Re_h
        # X2 = Fr
        # X3 = Fr_d
            
    sc_X = joblib.load("./ModelFiles/Scalers/sc_X3.pkl")
    X = CheckDimension(X)
    X = sc_X.transform(X)
    x0 = X[:,0]
    x1 = X[:,1]
    x2 = X[:,2]
    
    Fsus = 0.915636+-0.242087*((np.cos((np.sin(x2)+x0))/(((x0*x0)-(-3.678000-x0))-np.log((19.545000-x1)))))
            
            
    return Fsus


def GPGOMEA5(X):
    # Fsus equation that makes 
        # X1 = Re_h
        # X2 = Fr
        # X3 = Fr_d
    sc_X = joblib.load("./ModelFiles/Scalers/sc_X5.pkl")
    X = CheckDimension(X)
    X = sc_X.transform(X)
    x0 = X[:,0]
    x1 = X[:,1]
    x2 = X[:,2]
    x3 = X[:,3]
#    x4 = X[:,4]

    sin = np.sin
    cos = np.cos
#    tanh = np.tanh
#    exp = np.exp
    plog = np.log
    sqrt = np.sqrt

    Fsus = 0.706104+0.164698*((x2-(cos(((sqrt(x0)-(x3-x2))/(x2+1)))+sin(sin((plog(x1)-(x0*x2)))))))
    return Fsus



def SPUCI5(X):
    x = [0.1946199 , -0.04042124,  0.11720572, -0.120427]
    n,nvar = X.shape
    Wh = X[:,0]
#    dstar = X[:,1]
    Reh = X[:,2]
    Frd = X[:,3]
#    Rew = X[:,4]
    
    n,nvar = X.shape
    ypred = np.ones(n)
    ypred = ypred * x[0] * Wh**x[1] * Reh**x[2] * Frd**x[3]
    

    
    return ypred
    
def SPUCI3(X):
    x = [0.18465685,  0.11071018, -0.13309309]
    n,nvar = X.shape
    ReH = X[:,0]
#    Fr = X[:,1]
    Fr_d = X[:,2]
    
    ypred = np.ones(n)
    ypred = ypred * x[0] * ReH**x[1] * Fr_d**x[2]
    
    return ypred



def BPGP3(X):
    # Fsus equation that makes 
        # X1 = Re_h
        # X2 = Fr
        # X3 = Fr_d
    
    sc_X = joblib.load("./ModelFiles/Scalers/sc_X3.pkl")
    X = CheckDimension(X)
    X = sc_X.transform(X)
#    X1 = X[:,0]
#    X2 = X[:,1]
    X3 = X[:,2]
    
    sin = np.sin
    cos = np.cos
#    tanh = np.tanh
#    exp = np.exp
#    sqrt = np.sqrt
    plog = np.log1p

    Fsus = ((X3/18.291000)+(((((((sin((plog((X3/-15.583000))*(plog((X3+11.965000))/sin((sin((13.086000/X3))/cos(sin((16.146000-plog(X3)))))))))/plog(X3))/sin(((-3.545000+-22.344000)-sin(X3))))/(plog(X3)/((X3/X3)*(6.466000-(-22.224000*((cos(X3)/X3)/X3))))))-X3)/(X3/sin((X3+12.286000))))/(-10.503000-X3))+((cos(X3)/(-33.573000+X3))+(((X3/((X3+((cos((X3+(34.023000*X3)))*cos(X3))*X3))/sin(X3)))/18.291000)+((sin(plog(X3))/18.291000)+((sin(plog(X3))/18.291000)+((sin((X3-(((sin(X3)+X3)-cos(plog(X3)))+X3)))/18.291000)+(((X3/(X3+(X3+X3)))/18.291000)+((sin(X3)/(24.773000*X3))+((sin((((cos(X3)*-16.508000)-((plog(sin((-33.443000*X3)))*X3)+X3))-(X3+sin(X3))))/18.291000)+((sin((((X3-sin(X3))-X3)-(X3+plog(plog(X3)))))/(21.649000/plog(X3)))+((sin((((plog(sin(X3))/X3)*X3)/X3))/18.291000)+((sin((sin((X3*3.941000))-X3))/18.291000)+((sin((X3/X3))/18.291000)+((sin(((sin(X3)-X3)-((-33.457000-X3)-(-31.745000/X3))))/18.291000)+((sin(((X3/X3)*X3))/(plog(X3)+24.316000))+sin(sin(plog((10.992000/X3))))))))))))))))))))
 
    return Fsus
    
def BPGP5(X):
        # X1 = W/h
        # X2 = d_*
        # X3 = Re_h
        # X4 = Fr_d
        # X5 = Re_w

    sc_X = joblib.load("./ModelFiles/Scalers/sc_X5.pkl")
    X = CheckDimension(X)
    X = sc_X.transform(X)
        
#    X1 = X[:,0]
#    X2 = X[:,1]
#    X3 = X[:,2]
#    X4 = X[:,3]
    X5 = X[:,4]
    
    sin = np.sin
    cos = np.cos
#    tanh = np.tanh
#    exp = np.exp
#    sqrt = np.sqrt
    plog = np.log1p
    
    Fsus = cos(((((1.087969/((X5/(X5+(cos(plog(sin((X5/sin((((((cos(X5)*X5)/cos(44.077000))+plog(X5))-((39.203000-X5)+((X5+X5)-plog(X5))))*X5))))))-(X5*cos((X5*sin(((X5*(X5+cos(cos((X5*X5)))))*(sin(cos(X5))*X5)))))))))/plog((cos(((X5/((X5*(X5/cos((X5*cos(-29.274000)))))+X5))-X5))+X5))))/(((sin(((X5/(X5+plog(X5)))-plog(X5)))+X5)/X5)/X5))/sin(((((X5*((plog(((sin(X5)+(X5/(plog(plog(sin(X5)))-plog(((X5+((X5*X5)-X5))/(X5-X5))))))-plog(sin(X5))))-cos(X5))+X5))+(sin(plog((X5-sin(X5))))-X5))/((X5-(X5*sin((X5-X5))))-plog(plog(X5))))*X5)))/sin((sin(X5)*(sin(X5)+plog(sin(cos((cos((7.202000+(X5+(cos(X5)*cos((X5+X5))))))/cos(X5))))))))))
    
    return Fsus


def FsusPredict(X,Fsusmodel='Operon3'):
    
    
    if Fsusmodel == 'Operon3':
        Fsus = Operon3(X)
    elif Fsusmodel == 'Operon5':
        Fsus = Operon5(X)

        
    elif Fsusmodel == 'GPGOMEA3':
        Fsus = GPGOMEA3(X)    
    elif Fsusmodel == 'GPGOMEA5':
        Fsus = GPGOMEA5(X)

        
    elif Fsusmodel == 'BPGP3':
        Fsus = BPGP3(X)
    elif Fsusmodel == 'BPGP5':
        Fsus = BPGP5(X)

        
    elif Fsusmodel == 'MGGP3':
        Fsus = MGGP3(X)
    elif Fsusmodel == 'MGGP5':
        Fsus = MGGP5(X)

        
    elif Fsusmodel == 'SVR3':
        Fsus = SVR3(X)
    elif Fsusmodel == 'SVR5':
        Fsus = SVR5(X)
    
    elif Fsusmodel == 'SPUCI3':
        Fsus = SPUCI3(X)
    elif Fsusmodel == 'SPUCI5':
        Fsus = SPUCI5(X)
    
    
#    Fsus[Fsus<0] = 0.000001
    Fsus[Fsus<0] = 0.0001
    Fsus[Fsus>1] = 1
    
    return Fsus

def CsustoSL(Csus,Q, Cunit='mg/L', SLunit='t/d'):
    # It returns suspended load from the discharge 
    # Assumming the Csus is the bulk mean suspended load concentration
    # Csus: suspended sediment concentration [kg/m^3]
    # Q: water dishcarge [cms]
    # Cunit: concentration unit (default='mg/L' , options=['mg/L','kg/m^3'])
    # SLunit: SL unit (default='t/d' , options=['kg/s','t/d'])
    
    if Cunit == 'mg/L':
        Csus = Csus /1000
    elif Cunit == 'g/L':
        Csus = Csus
    elif Cunit == 'kg/m^3':
        Csus = Csus

    # by this transformation Csus unit becomes kg / m^3    
    
    if SLunit == 't/d':
        SL = Csus*Q *  86400 / 1000
    elif SLunit == 'kg/s':
        SL = Csus*Q 
    
    return SL

def FsustoTL(Fsus,SL):
    # This function transforms Fsus and suspended load (SL) to total load (TL)
    # Unit is preserved
    # Fsus: SL/ TL suspended load fraction to total load
    # SL: suspended load
    return SL/Fsus


def main(U,h,W = None , d50=None,S0=None,nu=10**-6, T=None, ustar = None, rhos = None,
          nv = 5,model = 'SVR'):
    # Please match the data size
    # Basic dimensions
    # d50: mm
    # T: Celsius degree -> For nu recalcuation
    # else: m or m/s
    
    if np.max(W == None):
        nv = 3
        print("No width -> n_var = 3")
    
    
    df = MakeDimLess(U, h, W=W , d50=d50,S0=S0,nu=nu, T=T, ustar = ustar, rhos = rhos)
    
    X =dftoX(df,n_var=nv) 
    
    if model == 'SVR':
    
        nanloc = np.isnan(X)
        infloc = np.isinf(X)
        
        X[nanloc] = 0
        
        nancol = np.max(nanloc,axis= 1)
        nancol = nancol + np.max(infloc,axis= 1)
    
    Fsus = FsusPredict(X,Fsusmodel=model + str(nv))
    
    if model == 'SVR':
        Fsus[nancol] = np.nan
        print('\nData has nan (or inf) components so that the corresponding result is converted to None components.\n')
    
    return Fsus



if __name__ == "__main__":
    
    #Manual input==========================================================
    
    U = [0.5, 2]   # m/s
    h = 0.5        # m
    W = 2          # m
    d50 = 1        # mm
    S0 = 0.001     # m/mg
    
    U = [0.422898,0.447341,0.388513,0.556208,0.514672,0.588146,0.572572]   # m/s
    h = [0.511879,0.384278,0.545732,0.411655,0.405924,0.332654,0.698791]        # m
    W = [34.4382,53.36925,30.98015,24.3145,23.19743,24.9305,11.5436]          # m
#    d50 = [1.164,1.146,1.386,2.494,1.781,1.578,1.376]        # mm
    d50 = [1.164069761,1.067942082,1.338841514,1.59536914,1.784448337,1.609665039,1.336452496] 
    S0 = 0.001     # m/m
    
    
    Cunit = 'mg/L' # Unit of Csus
    SLunit = 'kg/s'# 
    
    Csus = [2.9396,3.0952,3.9658,3.7381,6.8214,6.0929,31.4524] # ppm = mg/l - corresponded to Cunit
    
    model = 'SVR'  # Empirical model derivation method ('SVR','Operon','MGGP','GPGOMEA','BPGP')-str
    nv = 5         # Number of variables used to model derivation (3,5)-int
    
    #========================================================== Processing
    U = np.array(U)
    h = np.array(h)
    W = np.array(W)
    d50 = np.array(d50)
    S0 = np.array(S0)
    Csus = np.array(Csus)
    Q = U * W * h # cms

    Fsus = main(U,h,W = W, d50=d50,S0=S0,nu=10**-6, T=20, ustar = None, rhos = None,
          nv = nv,model = model)
    
    SL = CsustoSL(Csus,Q, Cunit=Cunit, SLunit=SLunit)
    
    TL = FsustoTL(Fsus,SL)
        
    
    #========================================================== Printing
    print('='*40)
    Invar = {'U (m/s)':U, 'h (m)':h, 'W (m)':W, 'd50 (mm)': d50, 'S0':S0, 'Q (cms)': Q}
    print("Input variables: ",Invar,sep='\n')
    
    print('='*40)
    print("Fsus model: ",model + str(nv),sep='\n')
    print("Fsus: ",Fsus,sep='\n')
    
    print('='*40)
    print("Csus ("+Cunit+"):",Csus,sep='\n')
    print("SL ("+SLunit+"):",SL,sep='\n')
    print("TL ("+SLunit+"):",TL,sep='\n')
    
    print('='*40)
    