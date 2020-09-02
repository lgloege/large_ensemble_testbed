import xarray as xr
import xarray.ufuncs as xu
import numpy as np

def wanninkhof92(T=None, 
                 S=None, 
                 P=None,
                 u_mean=None, 
                 u_var=None, 
                 xCO2=None,
                 pCO2_sw = None,
                 iceFrac = None,
                 scale_factor=0.27):
    '''
    Calculate air-sea CO2 flux following Wanninkhof 1992
    
    Inputs
    ============
    T = Temperature [degC]
    S = Salinity  [parts per thousand]
    P = Atmospheric pressure at 10m [atm]
    u_mean = mean wind speed [m/s]
    u_var = variance of wind speed m/s averaged monthly [m/s]
    xcO2 = atmoapheric mixing ratio of CO2 [ppm]
    pCO2_sw = CO2 partial pressure of seawter [uatm]
    scale_factor = gas transfer scale factor (default=0.27)
    
    Output
    ============
    F_co2 = air-sea co2 flux [molC/m2/yr]
    
    References
    ============
    Weiss (1974) [for solubility of CO2]
        Carbon dioxide in water and seawater: the solubility of a non-ideal gas
    Weiss and Price (1980) [for saturation vapor pressure, not used here]
        Nitrous oxide solubility in water and seawater     
    Dickson et al (2007) [for saturation vapor pressure]
        Guide to best practices for Ocean CO2 measurements, Chapter 5 section 3.2
    Wanninkhof (1992) [for gas transfer and Schmidt number]
        Relationship between wind speed and gas exchange over the ocean   
    Sweeney et al. (2007) [for gas transfer scale factor]
        Constraining global air‐sea gas exchange for CO2 with recent bomb 14C measurements
    Sarmiento and Gruber (2007) [for overview and discussion of CO2 flux]
        Ocean biogeochmiecal dynsmics, Ch3 see panel 3.2.1
    
    Notes
    ============
    this code is optimized to work with Xarray
    
    ## Notes on partial pressure moist air
    P_h20/P = saturation vapor pressure (Weiss and Price 1980)
    Water vapor is generally assumed to be at saturation in the vicinity of the air-sea interfae.
        p_moist = X_dry (P - P_h20)
                = x_dry * P (1-P_h20/P)
                = P_dry (1-P_h20/P)

    ## Notes on U2
    Need to add wind speed variance if using Wanninkhof (1992)
    If we don't, then we are underestimating the gas-transfer velocity
    U2 = (U_mean)^2 + (U_prime)^2, (U_prime)^2 is the standard deviation
    See Sarmiento and Gruber (2007) for a discussion of this
    
    ## Test values
    # T = 298.15
    # S = 35
    
    '''
    
    # ==================================================================
    # 0. Conversions
    # ==================================================================
    # Convert from Celsius to kelvin
    T_kelvin = T + 273.15
    
    
    # ==================================================================
    # 1. partial pressure of CO2 in moist air
    #    Units : uatm
    # ==================================================================    
    # Weiss and Price (1980) formula
    ## P_sat_vapor = xu.exp(24.4543 - 67.4509*(100/T_kelvin) - 4.8489*xu.log(T_kelvin/100) - 0.000544*S)
    
    
    # Following Dickson et al (2007) Chapter 5 section 3.2
    a1 = -7.85951783
    a2 = 1.84408259
    a3 = -11.7866497
    a4 = 22.6807411
    a5 = -15.9618719
    a6 = 1.80122502

    Tc = 647.096
    pc = (22.064 * 10**6) / 101325
    g = (1-(T_kelvin/Tc))

    # pure water saturation vapor pressure, Wagner and Pruß, 2002
    p_h20 = pc * xu.exp((Tc/T_kelvin) * (a1*g + 
                          a2*g**(1.5) + 
                          a3*g**(3) + 
                          a4*g**(3.5) + 
                          a5*g**(4) + 
                          a6*g**(7.5)))


    # total molality 
    MT = (31.998*S) / (1000 - 1.005*S)

    # osmotic coefficient at 25C by Millero (1974)
    b1 = 0.90799 
    b2 = - 0.08992
    b3 = 0.18458
    b4 = -0.07395
    b5 = -0.00221
    phi = b1 + b2*(0.5*MT) + b3*(0.5*MT)**2 + b4*(0.5*MT)**3 + b5*(0.5*MT)**4

    # vapor pressure above sea-water, Dickson et al. (2007), Chapter 5, section 3.2
    P_sat_vapor = p_h20 * xu.exp(-0.018*phi*MT)

    # Partial pressure of CO2 in moist air, Dickson et al. (2007), SOP 4 section 8.3
    P_moist = (P * xCO2) * (1 - P_sat_vapor)
    
    
    # ==================================================================
    # 2. Solubility of CO2
    #    Notes : T in degC, S in PSU, 
    #            1.0E-6 converts to correct to muatm
    #    Units : to mol.kg^{-1}uatm^{-1}
    #    Reference : Weiss (1974)
    # ==================================================================  
    S_co2 = xu.exp( 9345.17 / T_kelvin - 60.2409 + \
                     23.3585 * xu.log( T_kelvin / 100 ) + \
                     S * ( 0.023517 - 0.00023656 * T_kelvin + \
                          0.0047036 * ( T_kelvin / 100 )**2 )) *1.0E-6
    
    
    # ==================================================================
    # 3. Gas transfer velocity
    #    Units : cm/hr
    #    Reference : Wanninkhof (1992)
    #                Sweeney et al. (2007)
    #                per.comm with P. Landschutzer Feb. 19 2019
    # ==================================================================
    # Dimensionless Schmidt number (Sc)
    # References Wanninkhof 1992
    A = 2073.1 
    B = 125.62 
    C = 3.6276 
    D = 0.043219
    
    # Schmidt number 
    # units : dimensionless
    Sc = A - B*T + C*T**2 - D*T**3
    
    # Gas transfer velocity
    # References :  Wanninkhof 1992, 
    #               Sweeney et al. 2007 scale factor
    k_w = scale_factor * (Sc/660)**(-0.5) * (u_mean**2 + u_var)

    
    # ================================================
    # 4. air-sea CO2 exchange
    #    Units : mol/m2/yr
    #    Reference : Wanninkhof (1992)
    # ================================================  
    # Convert from cm*mol/hr/kg to mol/m2/yr
    conversion = (1000/100)*365*24
    
    # Air-sea CO2 flux 
    F_co2 = k_w * S_co2 * (1 - iceFrac) * (P_moist - pCO2_sw) * conversion
    
    return F_co2


###################################################
# Calculate flux 
###################################################


def calculate_flux(model=None, member=None, fl_u10_std=None, dir_out=None, ):
    '''
    calculate_flux(model=None, member=None)
    calculate CO2 flux following Wanninkhof (1992) for each member.
    
    Inputs
    ============
    model  = name of model (CESM, CanESM2, GFDL, MPI)
    member = member number
    fl_u10_std = file path to u10-STD netcdf file
    dir_out = directory to store output (do not put "/" at end)
    
    Output
    ============
    None. This stores the calculated flux as a NetCDF file in the {data_dir} location.
    
    Notes
    ============
    This uses the read_model2() function contained in _define_model_class.ipynb
    This uses the scaled version of Sweeney et al. (2007) used in 
    Landschutzer et al. (2016). Scale factor 1.014 is from person comm. with Peter
    '''
    ###======================================
    ### Define directories
    ###====================================== 
    #dir_raw = '/local/data/artemis/workspace/gloege/SOCAT-LE/data/raw'
    #dir_clean = '/local/data/artemis/workspace/gloege/SOCAT-LE/data/clean'
    
    ### ================================================
    ### Get file path
    ### ================================================
    if model.upper()=='CESM':
        mem = f'{member:0>3}'
    if model.upper()=='GFDL':
        mem = f'{member:0>2}'
    if model.upper()=='MPI':
        mem = f'{member:0>3}'
    if model.upper()=='CANESM2':
        mem = f'{member}'

    ### Load auxillary data
    ds_SST     = read_model2(model=model, member=mem, variable='SST')
    ds_SSS     = read_model2(model=model, member=mem, variable='SSS')
    ds_iceFrac = read_model2(model=model, member=mem, variable='iceFrac')
    ds_pATM    = read_model2(model=model, member=mem, variable='pATM')
    ds_XCO2    = read_model2(model=model, member=mem, variable='XCO2')
    ds_U10     = read_model2(model=model, member=mem, variable='U10')
    ds_pCO2    = read_model2(model=model, member=mem, variable='pCO2')
    ds_pCO2_SOMFFN = read_somffn(model=model, member=mem)
    
    ### Wind speed variance 
    #ds_U_std = xr.open_dataset(f'{dir_clean}/ERA_interim/ERAinterim_1x1_u10-std_1982-2016.nc', 
    #                           decode_times=False)
    ds_U_std = xr.open_dataset(f'{fl_u10_std}', decode_times=False)
    ###======================================
    ### Put data into xarray dataset
    ###======================================
    ds = xr.Dataset(
        {
        'SST':(['time','lat','lon'], ds_SST['SST'] ),
        'SSS':(['time','lat','lon'], ds_SSS['SSS']),
        'iceFrac':(['time','lat','lon'], ds_iceFrac['iceFrac'] ),
        'U10':(['time','lat','lon'], ds_U10['U10']),
        'U10_std':(['time','lat','lon'], ds_U_std['u10']),
        'Patm':(['time','lat','lon'], ds_pATM['pATM'] ),
        'xCO2':(['time'], ds_XCO2['XCO2']),
        'pCO2_member':(['time','lat','lon'], ds_pCO2['pCO2']),
        'pCO2_somffn':(['time','lat','lon'], ds_pCO2_SOMFFN['pco2']),
        },

        coords={
        'lat': (['lat'], ds_pCO2_SOMFFN['lat']),
        'lon': (['lon'], ds_pCO2_SOMFFN['lon']),
        'time': (['time'], ds_pCO2_SOMFFN['time'])
        })

    F_somffn = wanninkhof92(T=ds['SST'], 
                 S=ds['SSS'], 
                 P=ds['Patm'],
                 u_mean=ds['U10'], 
                 u_std=ds['U10_std'], 
                 xCO2=ds['xCO2'],
                 pCO2_sw = ds['pCO2_somffn'],
                 iceFrac = ds['iceFrac'],
                 scale_factor=0.27*1.014)
        
    F_member = wanninkhof92(T=ds['SST'], 
                 S=ds['SSS'], 
                 P=ds['Patm'],
                 u_mean=ds['U10'], 
                 u_std=ds['U10_std'], 
                 xCO2=ds['xCO2'],
                 pCO2_sw = ds['pCO2_member'],
                 iceFrac = ds['iceFrac'],
                 scale_factor=0.27*1.014)

    # Save output in file
    ds_out = xr.Dataset(
        {
        'F_member':(['time','lat','lon'], F_member), 
        'F_somffn':(['time','lat','lon'], F_somffn),       
        },

        coords={
        'lat': (['lat'], ds['lat']),
        'lon': (['lon'], ds['lon']),
        'time': (['time'], ds['time'])
        })
        
    # Save to netcdf
    ds_out.to_netcdf(f'{dir_out}/CO2_flux_{model}{mem}_SOMFFN.nc')