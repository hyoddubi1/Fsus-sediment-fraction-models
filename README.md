# Fsus-sediment-fraction-models
Python scripts of the derived Fsus estimation models from "A novel efficient method of estimating suspended total1 sediment load fraction in natural rivers"

## How to use?

```python
	from FsusModels.FsusModes import main, CsustSL, FsustoTL
    #Manual input==========================================================
    
    U = [0.5, 2]   # m/s
    h = 0.5        # m
    W = 2          # m
    d50 = 1        # mm
    S0 = 0.001     # m/mg

    Cunit = 'mg/L' # Unit of Csus
    SLunit = 'kg/s'# 
    
    Csus = [2.9396,3.0952] # ppm = mg/l - corresponded to Cunit
    
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
```
Or type your hydro-morphic variables into the "FsusModels.py" and run it.
```
cd FsusModels
python FsusModels.py
```
Then, the result of the example will be printed.