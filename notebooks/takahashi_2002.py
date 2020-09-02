
import numpy as np
import xarray as xr

class takahashi_2002():
    def __init__(self, pco2, sst):
        self.pco2 = pco2
        self.sst = sst
    def pco2_T(self, dim='time'):
        return (self.pco2.mean(dim=dim) * xr.ufuncs.exp(0.0423*(self.sst - self.sst.mean(dim=dim)))).transpose('time','lat','lon')

    def pco2_nonT(self, dim='time'):
        return self.pco2 * xr.ufuncs.exp(0.0423*(self.sst.mean(dim=dim) - self.sst))
    
    def Dpco2_temp(self, dim='time'):
        return self.pco2_T(dim=dim).max(dim=dim) - self.pco2_T(dim=dim).min(dim=dim)
    
    def Dpco2_bio(self, dim='time'):
        return self.pco2_nonT(dim=dim).max(dim=dim) - self.pco2_nonT(dim=dim).min(dim=dim)
    
    def T_minus_B(self, dim='time'):
        return self.Dpco2_temp(dim=dim) - self.Dpco2_bio(dim=dim)