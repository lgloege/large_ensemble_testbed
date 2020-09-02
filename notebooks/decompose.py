import xarray as xr
import numpy as np
import pandas as pd
from pandas.core.nanops import nanmean as pd_nanmean
import scipy
from scipy import stats
import statsmodels.nonparametric.smoothers_lowess
from rstl import STL as RSTL

## Future versions of pandas will require you to explicitly register matplotlib converters
#from pandas.plotting import register_matplotlib_converters
#register_matplotlib_converters()

#==============================================
# DETREND UFUNCT FUNCTION
#=============================================
def _detrend_ufunc(data):
    """
    detrend_ufunc : detrend along the first axis
    
    This takes numpy array and detrends along the first axis
    This is applied to xarray datasets using .apply_ufunc()
    """
        
    ### This adds an extra dimension if 1D
    ### Turns DataArray into numpy array
    if (len(data.shape)==1):
        data = np.expand_dims(data, axis=1)
        
    ### Get dimensions
    ndim0 = np.shape(data)[0]
    ndim1 = np.shape(data)[1]

    ### Allocate space to store data
    data_dt = np.ones((ndim0, ndim1))*np.NaN
    #slope = np.ones((ndim1))*np.NaN
    #intercept = np.ones((ndim1))*np.NaN

    ### Loop over the stacked dimension
    #for dim1 in tqdm(range(ndim1)):
    for dim1 in range(ndim1):  
        ### Mask is true if not a NaN
        mask = ~np.isnan(data[:, dim1])
        
        ### If the mask is all false
        ### We will skip that point
        if np.sum(mask)!=0:
            ### Make the abscissa dimension
            x = np.arange(len(data[mask, dim1]))
                        
            ### calculates linear least-squares regression
            reg = scipy.stats.linregress(x, data[mask, dim1])
            
            ### make a linear fit
            yfit = reg.intercept + reg.slope * x
            
            ### Save regression coefficients
            ##slope[dim1] = reg.slope
            ##intercept[dim1] = reg.intercept

            ### subtract linear trend
            data_dt[mask, dim1] = data[mask, dim1] - yfit
            ### Alternatively - You can make this if statement one line like this
            #data_dt[mask,dim1] = scipy.signal.detrend(data[mask, dim1], axis=0, type='linear')
            
    return data_dt

#==============================================
# DETREND FUNCTION
#=============================================
def detrend(data, dim=None):
    ### Get coordinate names
    coords = list(dict(data.coords).keys())

    ### Pop out the coordinate you want to detrend over
    coords.pop(coords.index(dim))

    ### stack the other dimensions
    data = data.stack({'z':coords})
    
    ## Apply detrend
    out = xr.apply_ufunc(_detrend_ufunc, data)
    
    ## Unstack it
    return out.unstack('z').transpose(dim, coords[0], coords[1])

#==============================================
# LOWESS UFUNC
#=============================================
def _lowess_ufunc(data, lo_pts=None, lo_delta=None, it=None):
    '''
    _lowess(data, lo_pts=None, lo_delta=None)
    
    LOWESS (Locally Weighted Scatterplot Smoothing)
    A lowess function that outs smoothed estimates of endog
    at the given exog values from points (exog, endog)

    Parameters
    ----------
    data: 1-D numpy array
        The y-values of the observed points
    lo_pts: float
        Between 0 and 1. The fraction of the data used
        when estimating each y-value.
    lo_delta: float
        Distance within which to use linear-interpolation
        instead of weighted regression.

    Returns
    -------
    out: ndarray, float
        The returned array is two-dimensional if return_sorted is True, and
        one dimensional if return_sorted is False.
        If return_sorted is True, then a numpy array with two columns. The
        first column contains the sorted x (exog) values and the second column
        the associated estimated y (endog) values.
        If return_sorted is False, then only the fitted values are returned,
        and the observations will be in the same order as the input arrays.

    References
    ----------
    Cleveland, W.S. (1979) "Robust Locally Weighted Regression
    and Smoothing Scatterplots". Journal of the American Statistical
    Association 74 (368): 829-836.
    '''
    # https://github.com/joaofig/pyloess another way
    ### Define lowess smoother
    lowess = statsmodels.nonparametric.smoothers_lowess.lowess
    
    ### If importing an xr.DataArray make numpy array
    if (type(data)==type(xr.DataArray([]))):
        data = data.values
        
    ### This adds an extra dimension if 1D
    ### Turns DataArray into numpy array
    if (len(data.shape)==1):
        data = np.expand_dims(data, axis=1)
        
    ### Get dimensions
    ndim0 = np.shape(data)[0]
    ndim1 = np.shape(data)[1]

    ### Allocate space to store data
    data_dt = np.ones((ndim0, ndim1))*np.NaN
    #slope = np.ones((ndim1))*np.NaN
    #intercept = np.ones((ndim1))*np.NaN

    ### Loop over the stacked dimension
    #for dim1 in tqdm(range(ndim1)):
    for dim1 in range(ndim1):  
        ### Mask is true if not a NaN
        mask = ~np.isnan(data[:, dim1])
        
        ### If the mask is all false
        ### We will skip that point
        if np.sum(mask)!=0:
            ### apply smoother with parameters 
            trend = lowess(data[:,dim1], 
                    np.array([x for x in range(len(data[:,dim1]))], dtype=np.float64),
                    frac=lo_pts / len(data[:,dim1]),
                    delta=lo_delta * len(data[:,dim1]),
                    it=it)
            
            ### subtract linear trend
            data_dt[mask, dim1] = trend[:,1]
            del trend

    return data_dt


#==============================================
# LOWESS FUNCTION
#=============================================
def lowess(data, dim=None, lo_pts=None, lo_delta=None, it=3):
    ### Get coordinate names
    coords = list(dict(data.coords).keys())

    ### Pop out the coordinate you want to detrend over
    coords.pop(coords.index(dim))

    ### stack the other dimensions
    data = data.stack({'z':coords})
    
    ## Apply detrend
    out = xr.apply_ufunc(_lowess_ufunc, data, lo_pts, lo_delta, it)
    
    ## Unstack it
    return out.unstack('z').transpose(dim, coords[0], coords[1])


#==============================================
# SEASONAL CYCLE UFUNC
#=============================================
def _seaonal_cyle_ufunc(data, period=None):
    '''
    _seasonal_cyle(data, period=None)
    
    calculates a repeating seasonal cycle
    
    Parameters
    ----------
    data: 1-D numpy array
        The y-values of the observed points
    period: float
        the period of the seasonal cycle. 
        This depends on the sampling frequency of your data
        if monthly, then it is 12 if daily then 365

    Returns
    -------
    out: ndarray, float
        returns repeating seasonal cycle
    '''        
    ### This adds an extra dimension if 1D
    ### Turns DataArray into numpy array
    if (len(data.shape)==1):
        data = np.expand_dims(data, axis=1)
       
    ### If importing an xr.DataArray make numpy array
    if (type(data)==type(xr.DataArray([]))):
        data = data.values
        
    ### Get dimensions
    ndim0 = np.shape(data)[0]
    ndim1 = np.shape(data)[1]

    ### Allocate space to store data
    seasonal = np.ones((ndim0, ndim1))*np.NaN

    ### Loop over the stacked dimension
    #for dim1 in tqdm(range(ndim1)):
    for dim1 in range(ndim1):  
        ### Mask is true if not a NaN
        mask = ~np.isnan(data[:, dim1])
        
        ### If the mask is all false
        ### We will skip that point
        if np.sum(mask)!=0:      
            period_averages = np.array([pd_nanmean(data[i::period,dim1]) for i in range(period)])
            period_averages = period_averages - np.mean(period_averages)
            seasonal[:,dim1] = np.tile(period_averages, 
                                       len(data[:,dim1]) // period + 1)[:len(data[:,dim1])] 
            
    return seasonal


#==============================================
# SEASONAL CYCLE
#=============================================
def seasonal_cycle(data, dim=None, period=None):
    ### Get coordinate names
    coords = list(dict(data.coords).keys())

    ### Pop out the coordinate you want to detrend over
    coords.pop(coords.index(dim))

    ### stack the other dimensions
    data = data.stack({'z':coords})
    
    ## Apply detrend
    out = xr.apply_ufunc(_seaonal_cyle_ufunc, data, period)
    
    ## Unstack it
    return out.unstack('z').transpose(dim, coords[0], coords[1])



#==============================================
# RSTL method uFUNC
#=============================================
def _STL_ufunc(data, fq=None, 
               s_window=None, s_degree=None, 
               t_window=None, t_degree=None, 
               l_window=None, l_degree=None):
    '''            
    _STL_ufunc(data, fq=None, 
               s_window=None, s_degree=None, 
               t_window=None, t_degree=None, 
               l_window=None, l_degree=None)
    
    Applies the STL method as defined in the R function
    https://stat.ethz.ch/R-manual/R-devel/library/stats/html/stl.html

    Parameters
    ----------
    fq:
        frequency of data 
    s_window:
        either the character string "periodic" or the span (in lags) of the 
        loess window for seasonal extraction 
    s_degree:
        degree of locally-fitted polynomial in seasonal extraction. Should be zero or one.


    t_window:
        the span (in lags) of the loess window for trend extraction, which should be odd.
    t_degree:
        degree of locally-fitted polynomial in trend extraction. Should be zero or one.
    l_window:
        the span (in lags) of the loess window of the low-pass filter used for each subseries. 
        Defaults to the smallest odd integer greater than or equal to frequency(x) 
        which is recommended since it prevents competition between the trend and seasonal components.
        If not an odd integer its given value is increased to the next odd one.

    l_degree:
        degree of locally-fitted polynomial for the subseries low-pass filter. Must be 0 or 1.

    Returns
    -------
    out: ndarray, tuple
        (trend, seasonal, remainder, remainder_low)

    References
    ----------
    Cleveland, W.S. (1979) "Robust Locally Weighted Regression
    and Smoothing Scatterplots". Journal of the American Statistical
    Association 74 (368): 829-836.
    '''
    # https://github.com/joaofig/pyloess another way
    ### Define lowess smoother
    #lowess = statsmodels.nonparametric.smoothers_lowess.lowess
    
    #data = data.values
    ### If importing an xr.DataArray make numpy array
    if (type(data)==type(xr.DataArray([]))):
        data = data.values
        
    ### This adds an extra dimension if 1D
    ### Turns DataArray into numpy array
    if (len(data.shape)==1):
        data = np.expand_dims(data, axis=1)
        
   # data=data.values
    
    ### Get dimensions
    ndim0 = np.shape(data)[0]
    ndim1 = np.shape(data)[1]

    ### Allocate space to store data
    trend = np.ones((ndim0, ndim1))*np.NaN
    seasonal = np.ones((ndim0, ndim1))*np.NaN
    remainder = np.ones((ndim0, ndim1))*np.NaN
    remainder_low = np.ones((ndim0, ndim1))*np.NaN
    output = np.ones((ndim0, ndim1, 4))*np.NaN

    ### Loop over the stacked dimension
    #for dim1 in tqdm(range(ndim1)):
    for dim1 in range(ndim1):  
        ### Mask is true if not a NaN
        mask = ~np.isnan(data[:, dim1])
        
        ### If the mask is all false
        ### We will skip that point
        if np.sum(mask)!=0:
            ### apply smoother with parameters 

           
            ## Apply STL method
            out = RSTL(data[:,dim1], fq, 
                         s_window=s_window, s_degree=s_degree, 
                         t_window=t_window, t_degree=t_degree, 
                         l_window=l_window, l_degree=l_degree )
                        
            ## Apply STL method again to get low frequency remainder signal
            out2 = RSTL(out.remainder, fq, 
                    s_window='periodic', s_degree=0, 
                    t_window=13, t_degree=1, 
                    l_window=13, l_degree=1 )
            
            ### subtract linear trend
            trend[mask, dim1] = out.trend
            seasonal[mask, dim1] = out.seasonal
            remainder[mask, dim1] = out.remainder
            remainder_low[mask, dim1] = out2.trend
            del out, out2
    return (trend, seasonal, remainder, remainder_low)



#==============================================
# RSTL method
#=============================================
def STL(data, dim=None, fq=None, 
               s_window=None, s_degree=None, 
               t_window=None, t_degree=None, 
               l_window=None, l_degree=None):
    '''            
    SRL(data, dim=None, fq=None, 
               s_window=None, s_degree=None, 
               t_window=None, t_degree=None, 
               l_window=None, l_degree=None)
    
    Applies the STL method as defined in the R function
    https://stat.ethz.ch/R-manual/R-devel/library/stats/html/stl.html

    Parameters
    ----------
    dim:
        dimension to apply STL over
    fq:
        frequency of data 
    s_window:
        either the character string "periodic" or the span (in lags) of the 
        loess window for seasonal extraction 
    s_degree:
        degree of locally-fitted polynomial in seasonal extraction. Should be zero or one.


    t_window:
        the span (in lags) of the loess window for trend extraction, which should be odd.
    t_degree:
        degree of locally-fitted polynomial in trend extraction. Should be zero or one.
    l_window:
        the span (in lags) of the loess window of the low-pass filter used for each subseries. 
        Defaults to the smallest odd integer greater than or equal to frequency(x) 
        which is recommended since it prevents competition between the trend and seasonal components.
        If not an odd integer its given value is increased to the next odd one.

    l_degree:
        degree of locally-fitted polynomial for the subseries low-pass filter. Must be 0 or 1.

    Returns
    -------
    out: xarray dataset with (trend, seasonal, remainder, remainder_low)

    References
    ----------
    Cleveland, W.S. (1979) "Robust Locally Weighted Regression
    and Smoothing Scatterplots". Journal of the American Statistical
    Association 74 (368): 829-836.
    '''
    
    ### Get coordinate names
    coords = list(dict(data.coords).keys())

    ### Pop out the coordinate you want to detrend over
    coords.pop(coords.index(dim))

    ### stack the other dimensions
    data = data.stack({'z':coords})
    
    ## Apply detrend
    out = xr.apply_ufunc(_STL_ufunc, data, fq, 
               s_window, s_degree, 
               t_window, t_degree, 
               l_window, l_degree, output_core_dims=[[], [], [], []])
    
    ## Unstack it
    output = xr.merge([out[0].unstack('z').transpose(dim, coords[0], coords[1]).rename('trend'),
            out[1].unstack('z').transpose(dim, coords[0], coords[1]).rename('seasonal'),
            out[2].unstack('z').transpose(dim, coords[0], coords[1]).rename('remainder'),
            out[3].unstack('z').transpose(dim, coords[0], coords[1]).rename('remainder_low')])
    
    return output.transpose(dim, coords[0], coords[1])



#==============================================
# STL trend Ufunc
#=============================================
def _STL_trend_ufunc(data, fq=None, 
               s_window=None, s_degree=None, 
               t_window=None, t_degree=None, 
               l_window=None, l_degree=None):
    '''            
    _STL_trend_ufunc(data, fq=None, 
               s_window=None, s_degree=None, 
               t_window=None, t_degree=None, 
               l_window=None, l_degree=None)
    
    Applies the STL trend method as defined in the R function
    https://stat.ethz.ch/R-manual/R-devel/library/stats/html/stl.html

    Parameters
    ----------
    fq:
        frequency of data 
    
    s_window:
        either the character string "periodic" or the span (in lags) of the 
        loess window for seasonal extraction 
        
    s_degree:
        degree of locally-fitted polynomial in seasonal extraction. Should be zero or one.

    t_window:
        the span (in lags) of the loess window for trend extraction, which should be odd.
    
    t_degree:
        degree of locally-fitted polynomial in trend extraction. Should be zero or one.
    
    l_window:
        the span (in lags) of the loess window of the low-pass filter used for each subseries. 
        Defaults to the smallest odd integer greater than or equal to frequency(x) 
        which is recommended since it prevents competition between the trend and seasonal components.
        If not an odd integer its given value is increased to the next odd one.

    l_degree:
        degree of locally-fitted polynomial for the subseries low-pass filter. Must be 0 or 1.

    Returns
    -------
    out: ndarray, tuple
        (trend, seasonal, remainder, remainder_low)

    References
    ----------
    Cleveland, W.S. (1979) "Robust Locally Weighted Regression
    and Smoothing Scatterplots". Journal of the American Statistical
    Association 74 (368): 829-836.
    '''
    # https://github.com/joaofig/pyloess another way
    ### Define lowess smoother
    #lowess = statsmodels.nonparametric.smoothers_lowess.lowess
    
    #data = data.values
    ### If importing an xr.DataArray make numpy array
    if (type(data)==type(xr.DataArray([]))):
        data = data.values
        
    ### This adds an extra dimension if 1D
    ### Turns DataArray into numpy array
    if (len(data.shape)==1):
        data = np.expand_dims(data, axis=1)
        
   # data=data.values
    
    ### Get dimensions
    ndim0 = np.shape(data)[0]
    ndim1 = np.shape(data)[1]

    ### Allocate space to store data
    trend = np.ones((ndim0, ndim1))*np.NaN

    ### Loop over the stacked dimension
    #for dim1 in tqdm(range(ndim1)):
    for dim1 in range(ndim1):  
        ### Mask is true if not a NaN
        mask = ~np.isnan(data[:, dim1])
        
        ### If the mask is all false
        ### We will skip that point
        if np.sum(mask)!=0:
            ### apply smoother with parameters 

           
            ## Apply STL method
            out = RSTL(data[:,dim1], fq, 
                         s_window=s_window, s_degree=s_degree, 
                         t_window=t_window, t_degree=t_degree, 
                         l_window=l_window, l_degree=l_degree )
                                    
            ### subtract linear trend
            trend[mask, dim1] = out.trend
            del out
    return trend

#==============================================
# STL trend
#=============================================
def STL_trend(data, dim=None, fq=None, 
               s_window=None, s_degree=None, 
               t_window=None, t_degree=None, 
               l_window=None, l_degree=None):
    '''            
    STL_trend(data, dim=None, fq=None, 
               s_window=None, s_degree=None, 
               t_window=None, t_degree=None, 
               l_window=None, l_degree=None)
    
    Applies the STL trend method as defined in the R function
    https://stat.ethz.ch/R-manual/R-devel/library/stats/html/stl.html

    Parameters
    ----------
    dim:
        dimension to apply STL_trend over
        
    fq:
        frequency of data 
    
    s_window:
        either the character string "periodic" or the span (in lags) of the 
        loess window for seasonal extraction 
        
    s_degree:
        degree of locally-fitted polynomial in seasonal extraction. Should be zero or one.

    t_window:
        the span (in lags) of the loess window for trend extraction, which should be odd.
    
    t_degree:
        degree of locally-fitted polynomial in trend extraction. Should be zero or one.
    
    l_window:
        the span (in lags) of the loess window of the low-pass filter used for each subseries. 
        Defaults to the smallest odd integer greater than or equal to frequency(x) 
        which is recommended since it prevents competition between the trend and seasonal components.
        If not an odd integer its given value is increased to the next odd one.

    l_degree:
        degree of locally-fitted polynomial for the subseries low-pass filter. Must be 0 or 1.

    Returns
    -------
    out: ndarray, tuple
        (trend, seasonal, remainder, remainder_low)

    References
    ----------
    Cleveland, W.S. (1979) "Robust Locally Weighted Regression
    and Smoothing Scatterplots". Journal of the American Statistical
    Association 74 (368): 829-836.
    '''
    ### Get coordinate names
    coords = list(dict(data.coords).keys())

    ### Pop out the coordinate you want to detrend over
    coords.pop(coords.index(dim))

    ### stack the other dimensions
    data = data.stack({'z':coords})
    
    ## Apply detrend
    out = xr.apply_ufunc(_STL_trend_ufunc, data, fq, 
               s_window, s_degree, 
               t_window, t_degree, 
               l_window, l_degree)
    
    ## Unstack it
    return out.unstack('z').transpose(dim, coords[0], coords[1])




def decompose_stl(da=None, 
                  var_name=None, 
                  mem=None, 
                  model=None, 
                  suffix=None,
                  dir_out=None,
                  lo_pts=12*10, 
                  lo_delta=0.01):
    '''
    decompose_stl(da=None, var_name=None)
        decompose a DataArray using STL method.
        
    Input
    ==========
    da : datarray
    var_name : name of variable for output
    mem : member number (eg. '001')
    model : model name (eg. 'CESM')
    suffix : ending suffix (eg. 'SOMFFM-float')
    dir_out : output directory (eg. './output')
    Output 
    ==========
    ds : dataset with original signal and decomposed
    
    Reference
    ==========
    Cleveland et al. 1990
    '''
    #dir_out = '/local/data/artemis/workspace/gloege/SOCAT-LE/data/clean/pCO2-float_decomp_stl'
    
    ###======================================
    ### STL Decomposition
    ###======================================
    ### 0. Load raw data
    data = da.copy()

    #print('detrend')
    ### 1. Detrend it 
    data_detrend = detrend(data, dim='time')

    #print('deseason')
    ### 2. Remove seasonal cycle
    # 2.1 -- seasonal cycle
    data_seasonal = seasonal_cycle(data_detrend, dim='time', period=12)

    #print('deseason')
    # 2.2 -- de-season the data
    data_deseason = data_detrend - data_seasonal

    #print('lowess')
    ### 3. calculate the LOWESS
    data_lowess = lowess(data_deseason, dim='time', lo_pts=lo_pts, lo_delta=lo_delta)

    #print('residual')
    ### 4. Residual term -- not explained by trend or seasonal cycle 
    data_residual = data_deseason - data_lowess

    ### Low frequency of residual
    #data_residual_low = stl.lowess(data_residual, dim='time', lo_pts=12, lo_delta=0.01)

    # Save data as dataset
    ###======================================
    ### Return dataset
    ###======================================
    ds_out = xr.Dataset(
        {    
        f'{var_name}': (['time', 'lat', 'lon'], data ),
        f'{var_name}_detrend': (['time', 'lat', 'lon'], data_detrend ),
        f'{var_name}_decadal': (['time','lat', 'lon'], data_lowess ),
        f'{var_name}_seasonal': (['time','lat', 'lon'], data_seasonal ),
        f'{var_name}_residual': (['time','lat', 'lon'], data_residual ),
        #f'{var_out}_residual_low': (['time','lat', 'lon'], data_residual_low ),
        },

        coords={
        'time': (['time'], ds['time']),
        'lat': (['lat'], ds['lat']),
        'lon': (['lon'], ds['lon']),
        })
    
    #return ds_out
    ds_out.to_netcdf(f'{dir_out}/{var_name}_decomp_{model}{mem}_{suffix}.nc')
    #ds_out.to_netcdf(f'{dir_out}/pco2_decomp_{model}{mem}_{model_or_recon}.nc') 