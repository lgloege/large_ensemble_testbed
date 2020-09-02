# ------------------------------------------------------------
# skill_metrics
# A collection of statistical functions optimized for xarray
# L. Gloege 2018
# ------------------------------------------------------------
# - std : standard deviation
# - corrleation : correlation
# - correlation_neff : correlation with effective sample size
# - avg_abs_error : average absolute error
# - avg_error : average error or bias
# - rmse : root-mean squared error
# - urmse : unbiased root-mean-squared error
# - ri : Reliability index
# - nse : Nash-Sutcliffe efficiency

import numpy as np
import xarray as xr
import scipy.stats as ss

class skill_metrics():
    #def covariance(x, y, dim='time'):
    # 
    #    return ((x - x.mean(dim=dim)) * (y - y.mean(dim=dim))).mean(dim=dim)
    
    
    def std(x, dim='time'):
        '''
        Computes standard deviation across speficied dimension

        Parameters
        ----------
        x : xarray DataArray
        dim : dimension to perform standard deviation over (default='time')
        
        Returns
        -------
        std : standard deviation across specified dimensions
        '''
        return x.std(dim=dim)
    

    def correlation(x, y, dim='time'):  
        '''
        Computes Pearson correlation coefficient across dimensions.
        The correlation coefficient, r, measures the tendency of the
        predicted and observed values to vary together

        Parameters
        ----------
        x : xarray DataArray
        y : xarray DataArray
        dim : dimension to perform correlation over (default='time')
        
        Returns
        -------
        r : Pearson correlation across dimension
        
        References:
        ---------- 
        1. Stow, C.A., et al. "Skill assessment for couples biological/physical 
        models of marine systems." J. Mar.Sys. 76.4 (2009).
        '''
        # kludgy fix when dim values differ for x and y
        # For example, when calculating auto-correlation
        if (x[dim].equals(y[dim])==False):
            print(f'the {dim} dim do not agree. I am setting y[{dim}]=x[{dim}]')
            y[dim] = x[dim]

        xmean = x.mean(dim=dim)
        ymean = y.mean(dim=dim)

        x_minus_mean = x - xmean
        y_minus_mean = y - ymean

        x_minus_mean2 = x_minus_mean**2
        y_minus_mean2 = y_minus_mean**2

        ssx = xr.ufuncs.sqrt((x_minus_mean2).sum(dim=dim))
        ssy = xr.ufuncs.sqrt((y_minus_mean2).sum(dim=dim))

        num = ( x_minus_mean * y_minus_mean ).sum(dim=dim)
        denom = ssx * ssy

        r=num/denom
        return r
    
    
    def correlation_neff(x, y, dim='time', lag=1, two_sided=True):
        '''
        Computes the Pearson product-moment coefficient of linear correlation.
        Effective sample size is taken into account. 

        Parameters
        ----------
        x : xarray DataArray
        y : xarray DataArray
        dim : dimension to perform correlation over (default='time')
        lag : lag for autocorrelation (default=1)
        two_sided : boolean (optional); Whether or not the t-test should be two sided.
        
        Returns
        -------
        r     : Pearson correlation across dimension
        p     : p-value for significance (significant if p<0.05)
        n_eff : effective degrees of freedom
        
        References:
        ---------- 
        1. Stow, C.A., et al. "Skill assessment for couples biological/physical 
        models of marine systems." J. Mar.Sys. 76.4 (2009).
        2. Wilks, Daniel S. Statistical methods in the atmospheric sciences. 
        Vol. 100. Academic press, 2011.
        3. Lovenduski, Nicole S., and Nicolas Gruber. "Impact of the Southern Annular Mode 
        on Southern Ocean circulation and biology." Geophysical Research Letters 32.11 (2005).
        '''
        # correlation between x and y
        r = skill_metrics.correlation(x, y, dim=dim)

        # number samples 
        n = len(x)

        # auto correlations
        xauto = skill_metrics.correlation(x[lag:,:,:], x[:-lag,:,:], dim=dim)
        yauto = skill_metrics.correlation(y[lag:,:,:], y[:-lag,:,:], dim=dim)

        # n_effective
        n_eff = n * (1 - xauto*yauto)/(1 + xauto*yauto)
        n_eff = xr.ufuncs.floor(n_eff)

        # Compute t-statistic.
        t = r * xr.ufuncs.sqrt((n_eff - 2)/(1 - r**2))

        # Propoability. if p<0.05, then statistically signfiicant
        if two_sided:
            p = xr.apply_ufunc(ss.t.sf, xr.ufuncs.fabs(t), n_eff-1)*2
        else:
            p =xr.apply_ufunc(ss.t.sf, xr.ufuncs.fabs(t), n_eff-1)

        return r, p, n_eff
    

    # ===========================================================
    # bias, AAE, and RMSE measure model prediction accuracy
    # Values near zero indicate a close match.
    # ===========================================================
    
    
    def avg_error(obs ,prd, dim='time'):
        '''
        average error or bias. 
        The average error is a measure of aggregate model bias, though values near zero 
        can be misleading because negative and positive discrepancies can cancel each other.

        Parameters
        ----------
        obs : xarray DataArray; observed variable
        prd : xarray DataArray; predicted variable
        dim : dimension to perform metric over (default='time')
        
        Returns
        -------
        bias : average error across dimension
        
        References:
        ---------- 
        1. Stow, C.A., et al. "Skill assessment for couples biological/physical 
        models of marine systems." J. Mar.Sys. 76.4 (2009).
        '''
        return (prd-obs).mean(dim=dim) 

    def avg_abs_error(obs, prd, dim='time'):
        '''
        average absolut error across dimension. 
        This measure how "off" any given prediction is on average

        Parameters
        ----------
        obs : xarray DataArray; observed variable
        prd : xarray DataArray; predicted variable
        dim : dimension to perform metric over (default='time')
        
        Returns
        -------
        aae : average absolute error across dimension
        
        References:
        ---------- 
        1. Stow, C.A., et al. "Skill assessment for couples biological/physical 
        models of marine systems." J. Mar.Sys. 76.4 (2009).
        '''
        return xr.ufuncs.fabs(prd-obs).mean(dim=dim) 

    
    def amp_ratio(obs, prd, dim='time'):
        '''
        ratio of the max-minus-min (predicted over observed) 
        This is a measure of the amplitude of the signal.
        similar to the normalized standard deviation

        Parameters
        ----------
        obs : xarray DataArray; observed variable
        prd : xarray DataArray; predicted variable
        dim : dimension to perform metric over (default='time')
        
        Returns
        -------
        amplitude : maximum minus the minimum
        
        References:
        ---------- 
        1. Stow, C.A., et al. "Skill assessment for couples biological/physical 
        models of marine systems." J. Mar.Sys. 76.4 (2009).
        '''
        return (((prd.max(dim=dim)-prd.min(dim=dim)) / (obs.max(dim=dim)-obs.min(dim=dim))) - 1)
    
    
    def rmse(m, r):
        '''
        root mean squared error. 

        Parameters
        ----------
        m  : xarray DataArray; observed variable
        r : xarray DataArray; predicted variable
        dim : dimension to perform metric over (default='time')
        
        Returns
        -------
        rmse : root-mean-squared error
        
        References:
        ---------- 
        1. Stow, C.A., et al. "Skill assessment for couples biological/physical 
        models of marine systems." J. Mar.Sys. 76.4 (2009).
        '''
        return xr.ufuncs.sqrt(xr.ufuncs.square((m-r)).mean(dim='time'))

    
    def std_star(obs, prd, dim='time'):
        '''
        normalized standard deviation.
        This measure how well the predicted variance agrees with observed

        Parameters
        ----------
        obs : xarray DataArray; observed variable
        prd : xarray DataArray; predicted variable
        dim : dimension to perform metric over (default='time')
        
        Returns
        -------
        std_star : normalized STD across dimension
        
        References:
        ---------- 
        1. Stow, C.A., et al. "Skill assessment for couples biological/physical 
        models of marine systems." J. Mar.Sys. 76.4 (2009).
        '''
        std_star = ((prd.std(dim=dim) / obs.std(dim=dim)) - 1)
        return std_star


    def urmse(m, r):
        '''
        unbiased root mean squared error

        Parameters
        ----------
        m  : xarray DataArray; observed variable
        r : xarray DataArray; predicted variable
        dim : dimension to perform metric over (default='time')
        
        Returns
        -------
        urmse : unbiased root-mean-squared error
        
        References:
        ---------- 
        1. Jolliff, J.K., et al. "Summary diagrams for coupled hydrodynamic-ecosystem model
        skill assessment." J. Mar.Sys. 76.64 (2009).
        '''
        return xr.ufuncs.sqrt(xr.ufuncs.square( (m - m.mean(dim='time')) - (r - r.mean(dim='time')) ).mean(dim='time'))
    
    
    def ri(m, r,dim='time'):
        '''
        reliability index.
        The reliability index quantifies the average factor by which model predictions 
        differ from observations. For example, an RI of 2.0 indicates that a model
        predicts the observations within a multiplicative factor of two, on average.

        Parameters
        ----------
        m  : xarray DataArray; observed variable
        r : xarray DataArray; predicted variable
        dim : dimension to perform metric over (default='time')
        
        Returns
        -------
        ri : reliability index
        
        References:
        ---------- 
        1. Stow, C.A., et al. "Skill assessment for couples biological/physical 
        models of marine systems." J. Mar.Sys. 76.4 (2009).
        2. Leggett and Williams. "A reliability index for models." 
        Ecological Modelling 13, 303–312 (1981).
        '''
        return xr.ufuncs.exp(xr.ufuncs.sqrt( xr.ufuncs.square( xr.ufuncs.log(m/r) ).mean(dim=dim)))
    
    
    def nse(m, r, dim='time'):
        '''
        Nash-Sutcliffe efficiency.
        The modeling efficiency measures how well a model predicts relative to 
        the average of the observations. It is related to the RMSE: 
                        MEF = 1 − [(RMSE)^2/s^2]
        where s^2 is the variance of the observations.

        Parameters
        ----------
        m  : xarray DataArray; observed variable
        r : xarray DataArray; predicted variable
        dim : dimension to perform metric over (default='time')
        
        Returns
        -------
        nse : nash-sutcliffe modeling efficiency 
        
        References:
        ---------- 
        1. Stow, C.A., et al. "Skill assessment for couples biological/physical 
        models of marine systems." J. Mar.Sys. 76.4 (2009).
        2. Loague, K., Green, R.E. "Statistical and graphical methods for evaluating
        solute transportmodels: overviewand application."  J. Contam.Hydro. 7, 51–73 (1991).
        3. Nash, J.E., Sutcliffe, J.V. "River flowforecasting through conceptual models,
        part 1 — a discussion of principles." J.Hydro. 10, 282–290 (1970).
        '''
        numer = xr.ufuncs.square(m - m.mean(dim=dim)).mean(dim=dim) - xr.ufuncs.square(r - m).mean(dim=dim)
        return numer / xr.ufuncs.square(m - m.mean(dim=dim)).mean(dim=dim)