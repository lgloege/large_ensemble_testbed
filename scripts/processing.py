
import numpy as np
import xarray as xr
import scipy.signal
import scipy.stats as stats

class processing():
    def __init__(self, data):
        #self._data_raw = data.copy()
        self._data = data.copy()
        
    @property
    def values(self):
        return self._data
    
    def rolling_mean(self, window=12):
        """
        rolling_mean(self, window=12)
        running mean centered in the window
        """
        self._data = self._data.rolling(time=window, center=True).mean()
        return self
  
    def detrend_ufunc(self, y):
        """
        This only works with 3D matrices
        """
        ### Get dimensions
        ndim0 = np.shape(y)[0]
        ndim1 = np.shape(y)[1]
        ndim2 = np.shape(y)[2]

        ### Allocate space to store data
        y_dt = np.ones((ndim0, ndim1, ndim2))*np.NaN
        #slope = np.ones((ndim1, ndim2))*np.NaN
        #intercept = np.ones((ndim1, ndim2))*np.NaN

        ### x vector
        x = np.arange(ndim0)

        ### Remove linear trend
        for dim1 in range(ndim1):
            for dim2 in range(ndim2):
                ### only proceed if no NaNs
                if(np.sum(np.isnan(y[:, dim1, dim2]))==0):
                    ### fit linear regression
                    reg = stats.linregress(x, y[:, dim1, dim2])
                    #y_dt[:,dim1, dim2] = scipy.signal.detrend(X[mask], axis=0, type='linear')
                    ### make predictions at x values
                    yfit = reg.intercept + reg.slope * x

                    ### Save regression coefficients
                    #slope[dim1, dim2] = reg.slope
                    #intercept[dim1, dim2] = reg.intercept

                    ### subtract linear trend
                    y_dt[:, dim1, dim2] = y[:, dim1, dim2] - yfit
                    
        return y_dt

    
    def detrend(self):
        self._data = xr.apply_ufunc(self.detrend_ufunc, self._data)
        return self


    def detrend4D_ufunc(self, y):
        """
        This only works with 4D matrices
        and assumes time is the second variable
        """
        ### Get dimensions
        ndim0 = np.shape(y)[0]
        ndim1 = np.shape(y)[1]
        ndim2 = np.shape(y)[2]
        ndim3 = np.shape(y)[3]

        ### Allocate space to store data
        y_dt = np.ones((ndim0, ndim1, ndim2, ndim3))*np.NaN

        ### x vector
        x = np.arange(ndim1)

        ### Remove linear trend
        for dim0 in range(ndim0):
            for dim2 in range(ndim2):
                for dim3 in range(ndim3):
                    ### only proceed if no NaNs
                    if(np.sum(np.isnan(y[dim0,:, dim2, dim3]))==0):
                        ### fit linear regression
                        reg = stats.linregress(x, y[dim0,:, dim2, dim3])
                        ### make predictions at x values
                        yfit = reg.intercept + reg.slope * x
                    
                        ### subtract linear trend
                        y_dt[dim0,:, dim2, dim3] = y[dim0,:, dim2, dim3] - yfit

        return y_dt

    def detrend4D(self):
        self._data = xr.apply_ufunc(self.detrend4D_ufunc, self._data)
        return self
    
    #def detrend_ufunc(self, X, axis=0):
    #    ### mask nan points
    #    mask = ~np.isnan(X)
    #    ## define output matrix
    #    out = X*np.nan
    #    ### detrend along axis
    #    out[mask] = scipy.signal.detrend(X[mask], axis=axis, type='linear')
    #    return out

    #def detrend(self,axis=0):
    #    self._data = xr.apply_ufunc(self.detrend_ufunc, self._data)
    #    return self

    
    def get_slope_ufunc(self, y):
        """
        This only works with 3D matrices
        """
        ### Get dimensions
        ndim0 = np.shape(y)[0]
        ndim1 = np.shape(y)[1]
        ndim2 = np.shape(y)[2]

        ### Allocate space to store data
        y_dt = np.ones((ndim0, ndim1, ndim2))*np.NaN
        #slope = np.ones((ndim1, ndim2))*np.NaN
        #intercept = np.ones((ndim1, ndim2))*np.NaN

        ### x vector
        x = np.arange(ndim0)

        ### Remove linear trend
        for dim1 in range(ndim1):
            for dim2 in range(ndim2):
                ### only proceed if no NaNs
                if(np.sum(np.isnan(y[:, dim1, dim2]))==0):
                    ### fit linear regression
                    reg = stats.linregress(x, y[:, dim1, dim2])
                    #y_dt[:,dim1, dim2] = scipy.signal.detrend(X[mask], axis=0, type='linear')
                    ### make predictions at x values
                    yfit = reg.intercept + reg.slope * x

                    ### Save regression coefficients
                    slope[dim1, dim2] = reg.slope
                    #intercept[dim1, dim2] = reg.intercept

                    ### subtract linear trend
                    #y_dt[:, dim1, dim2] = y[:, dim1, dim2] - yfit
                    
        return slope
    
    def get_slope(self):
        self._data = xr.apply_ufunc(self.get_slope_ufunc, self._data)
        return self
    
    def long_term_mean(self, dim='time'):
        '''long term mean alont dimension dim'''
        self._data = self._data.mean(dim)
        return self

    def global_avg(self, dim=['lat','lon']):
        '''long term mean alont dimension dim'''
        self._data = self._data.mean(dim)
        return self

    def global_mean(self, dim=['lat','lon']):
        '''long term mean alont dimension dim'''
        self._data = self._data.mean(dim)
        return self
    
    def global_median(self, dim=['lat','lon']):
        '''long term mean alont dimension dim'''
        self._data = self._data.median(dim)
        return self

    def zonal_mean(self,dim='lon'):
        '''long term mean alont dimension dim'''
        self._data = self._data.mean(dim)
        return self
    
    def zonal_median(self,dim='lon'):
        '''long term mean alont dimension dim'''
        self._data = self._data.median(dim)
        return self

    def ensemble_mean(self, dim='ensemble'):
        '''long term mean alont dimension dim'''
        self._data = self._data.mean(dim)
        return self

    def annual_mean(self, dim='time', nyears=35):
        self._data = self._data.groupby_bins(dim, nyears).mean(dim=dim)
        return self

    def remove_mean(self, dim='time'):
        ''' 
        remove_mean(X, dim='time')
        * use with .groupby_bins().apply() to remove annual mean
        '''
        self._data =  self._data - self._data.mean(dim)
        return self

    def annual_mean_repeating(self, dim='time', nyears=35, axis=0):
        tmp = self._data.groupby_bins(dim, nyears).mean(dim=dim)
        self._data = xr.DataArray(np.repeat(tmp.values, 12, axis=axis), dims=['time','lat','lon'])
        return self