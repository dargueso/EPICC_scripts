#!/usr/bin/env python
"""
#####################################################################
# Author: Daniel Argueso <daniel> @ UIB
# Date:   2017-11-28T18:25:06+01:00
# Email:  d.argueso@uib.es
# Last modified by:   daniel
# Last modified time: 2017-11-28T18:25:21+01:00
#
# @Project@ REHIPRE
# Version: x.0 (Beta)
# Description: Functions to calculate diagnostics from WRF outputs
#
# Dependencies: wrf-python, netCDF4, numpy
#
# Files:
#
#####################################################################
"""
import xarray as xr
import netCDF4 as nc
import numpy as np
import datetime as dt
import wrf as wrf
from atmopy.constants import const as const
from scipy.ndimage import gaussian_filter

#wrf.set_cache_size(0)
wrf.disable_xarray()

def compute_WRFvar (filename,varname,inputinf=None):
    """ Function to compute a variable name from wrf outputs
        filename: wrfout (or other file name used as input)
        varname : variable to be extracted or computed from WRF outputs
    """

    ncfile = nc.Dataset(filename,'r')



    if varname in list(ncfile.variables.keys()):
        varval = ncfile.variables[varname][:]
        varatt={}
        for att in ncfile.variables[varname].ncattrs():
            varatt[att] = getattr(ncfile.variables[varname],att)

    else:
        #Old version when is out of the functtion
        #compute=getattr('compute_'+varname)
        method_name='compute_%s' %(varname)
        possibles = globals().copy()
        possibles.update(locals())
        compute=possibles.get(method_name)

        varval, varatt=compute(filename,inputinf)

    ncfile.close()

    return varval,varatt


###########################################################
###########################################################

def compute_div_dx(u,v,dx,dy):

      """Function to calculate wind divergence providing the distance between grid points only
         u:zonal wind [m s-1]
         v:meridional wind [m s-1]
         dx: distance between gridpoints longitudinal [m]
         dy: distance between gridpoints latitudinal [m]
         ---
         dv: divergence [s-1]

         Author: Daniel Argueso @ CCRC, UNSW. Sydney (Australia)
         Created: Mon Mar  9 11:42:51 AEDT 2015 based on compute_div

      """

      dv = np.zeros(u.shape,dtype=np.float64)

      dv[...,1:-1,1:-1] = (u[...,1:-1,2:]-u[...,1:-1,:-2])/(2*dx) + (v[...,2:,1:-1]-v[...,:-2,1:-1])/(2*dy)

      return dv

###########################################################
###########################################################

def create_netcdf(var,filename):
    print((('\n Create output file %s') %(filename)))

    otimes = var['times']
    outfile = nc.Dataset(filename,'w',format='NETCDF4_CLASSIC',zlib=True, complevel=5)

    outfile.createDimension('time',None)
    outfile.createDimension('bnds',2)
    if var['values'].ndim == 4:

        outfile.createDimension('y',var['values'].shape[2])
        outfile.createDimension('x',var['values'].shape[3])
        outfile.createDimension('lev',var['values'].shape[1])

        outvar  = outfile.createVariable(var['varname'],'f',('time','lev','y','x'),fill_value=const.missingval)

    if var['values'].ndim == 3:
        outfile.createDimension('y',var['values'].shape[1])
        outfile.createDimension('x',var['values'].shape[2])

        outvar  = outfile.createVariable(var['varname'],'f',('time','y','x'),fill_value=const.missingval)

    outtime = outfile.createVariable('time','f8','time',fill_value=const.missingval)
    outtime_bnds = outfile.createVariable('time_bnds','f8',('time','bnds'),fill_value=const.missingval)
    outlat  = outfile.createVariable('lat','f',('y','x'),fill_value=const.missingval)
    outlon  = outfile.createVariable('lon','f',('y','x'),fill_value=const.missingval)

    if var['values'].ndim == 4:
        outlev = outfile.createVariable('levels','f',('lev'),fill_value=const.missingval)
        if var['varname']=='cloudfrac':
            setattr(outlev,"standard_name","cloud-level")
            setattr(outlev,"long_name","Clouds level")
            setattr(outlev,"units","")
            setattr(outlev,"_CoordinateAxisType","z")
        else:
            setattr(outlev,"standard_name","model-level")
            setattr(outlev,"long_name","Model level")
            setattr(outlev,"units","eta levels")
            setattr(outlev,"_CoordinateAxisType","z")

    setattr(outlat,"standard_name","latitude")
    setattr(outlat,"long_name","Latitude")
    setattr(outlat,"units","degrees_north")
    setattr(outlat,"_CoordinateAxisType","Lat")


    setattr(outlon,"standard_name","longitude")
    setattr(outlon,"long_name","Longitude")
    setattr(outlon,"units","degrees_east")
    setattr(outlon,"_CoordinateAxisType","Lon")

    setattr(outtime,"standard_name","time")
    setattr(outtime,"long_name","Time")
    setattr(outtime,"units","hours since 1949-12-01 00:00:00")
    setattr(outtime,"calendar","standard")

    setattr(outtime_bnds,"standard_name","time_bnds")
    setattr(outtime_bnds,"long_name","time_bounds")
    setattr(outtime_bnds,"units","hours since 1949-12-01 00:00:00")
    setattr(outtime_bnds,"calendar","standard")


    if len(otimes)==1:
        ## HARD CODED AT 1-HOUR
        step_seconds=3600.
    else:
        step_seconds = np.int((otimes[1]-otimes[0]).total_seconds())

    outtime[:] = nc.date2num([otimes[x] for x in range(len(otimes))],units='hours since 1949-12-01 00:00:00',calendar='standard')

    outtime_bnds[:,0]=nc.date2num([otimes[x]-dt.timedelta(seconds=step_seconds/2.) for x in range(len(otimes))],units='hours since 1949-12-01 00:00:00',calendar='standard')
    outtime_bnds[:,1]=nc.date2num([otimes[x]+dt.timedelta(seconds=step_seconds/2.) for x in range(len(otimes))],units='hours since 1949-12-01 00:00:00',calendar='standard')


    outlat[:]  = var['lat'][:]
    outlon[:]  = var['lon'][:]

    outvar[:] = var['values'][:]

    for outatt in list(var['atts'].keys()):
        setattr(outvar,outatt,var['atts'][outatt])

    setattr(outfile,"creation_date",dt.datetime.today().strftime('%Y-%m-%d'))
    setattr(outfile,'author','Daniel Argueso @UIB')
    setattr(outfile,'contact','d.argueso@uib.es')
    #setattr(outfile,'comments','files created from wrf outputs %s/%s' %(path_in,patt))

    outfile.close()

###########################################################
###########################################################

def compute_PR(filename, inputinf=None):
    """Function to calculate precipitation flux from a wrf output
       It also provides variable attribute CF-Standard
    """

    wrfvames = ['PREC_ACC_NC','PREC_ACC_C']
    varvals = {}
    ncfile = nc.Dataset(filename,'r')

    ## Specific to PR
    if hasattr(ncfile,'PREC_ACC_DT'):
        accum_dt = getattr(ncfile,'PREC_ACC_DT')
    else:
        print(("NO PREC_ACC_DT in input file. Set to default %s min" %(inputinf['acc_dt'])))
        accum_dt = int(inputinf['acc_dt'])

    ## Extracting variables required for diagnostic
    for var in wrfvames:
        varvals[var] = ncfile.variables[var][:]

    ## Computing diagnostic
    pr_acc = varvals['PREC_ACC_NC']+varvals['PREC_ACC_C']

    ## Deacumulating over prac_acc_dt (namelist entry)
    pr = pr_acc/(accum_dt*60.)



    atts = {"standard_name": "precipitation_flux",
                    "long_name"    : "total precipitation flux",
                    "units"        : "kg m-2 s-1"}

    ncfile.close()

    return pr,atts

def compute_PRNC(filename, inputinf=None):
    """Function to calculate non-convective precipitation flux from a wrf output
       It also provides variable attribute CF-Standard
    """

    ncfile = nc.Dataset(filename,'r')

    ## Specific to PR
    if hasattr(ncfile,'PREC_ACC_DT'):
        accum_dt = getattr(ncfile,'PREC_ACC_DT')
    else:
        print(("NO PREC_ACC_DT in input file. Set to default %s min" %(inputinf['acc_dt'])))
        accum_dt = int(inputinf['acc_dt'])


    ## Computing diagnostic
    prnc_acc = ncfile.variables['PREC_ACC_NC'][:]

    ## Deacumulating over prac_acc_dt (namelist entry)
    prnc = prnc_acc/(accum_dt*60.)



    atts = {"standard_name": "non_convective_precipitation_flux",
                    "long_name"    : "non-convective total precipitation flux",
                    "units"        : "kg m-2 s-1"}

    ncfile.close()

    return prnc,atts

def compute_PRCC(filename, inputinf=None):
    """Function to calculate nconvective precipitation flux from a wrf output
       that uses convective parameterization
       It also provides variable attribute CF-Standard
    """

    ncfile = nc.Dataset(filename,'r')

    ## Specific to PR
    if hasattr(ncfile,'PREC_ACC_DT'):
        accum_dt = getattr(ncfile,'PREC_ACC_DT')
    else:
        print(("NO PREC_ACC_DT in input file. Set to default %s min" %(inputinf['acc_dt'])))
        accum_dt = int(inputinf['acc_dt'][:])


    ## Computing diagnostic
    prnc_acc = ncfile.variables['PREC_ACC_C'][:]

    ## Deacumulating over prac_acc_dt (namelist entry)
    prnc = prnc_acc/(accum_dt*60.)



    atts = {"standard_name": "convective_precipitation_flux",
                    "long_name"    : "convective total precipitation flux",
                    "units"        : "kg m-2 s-1"}

    ncfile.close()

    return prnc,atts

def compute_PRACC(filename,inputinf=None):
    """Function to calculate precipitation flux from a wrf output
       It also provides variable attribute CF-Standard
    """

    wrfvames = ['RAINNC','RAINC']
    varvals = {}
    ncfile = nc.Dataset(filename,'r')

    ## Extracting variables required for diagnostic
    for var in wrfvames:
        varvals[var] = ncfile.variables[var][:]

    ## Computing diagnostic
    pracc = varvals['RAINC']+varvals['RAINNC']

    atts = {"standard_name": "precipitation_accumulated",
                    "long_name"    : "total accumulated precipitation",
                    "units"        : "kg m-2"}

    ncfile.close()

    return pracc,atts

def compute_TAS(filename,inputinf=None):
    """ Function to calculate 2-m temperature from WRF OUTPUTS
        It also provides variable attributes CF-Standard
    """

    ncfile = nc.Dataset(filename,'r')

    t2 = ncfile.variables['T2'][:]

    atts = {"standard_name": "air_temperature",
            "long_name":  "Surface air temperature",
            "units"    :  "K"                      ,
            "hgt"       :  "2 m"                    ,
            }

    ncfile.close()

    return t2,atts

def compute_TD2(filename,inputinf=None):
    """ Function to calculate 2-m dewpoint temperature from WRF OUTPUTS
        It also provides variable attributes CF-Standard
    """

    ncfile = nc.Dataset(filename,'r')

    td2 = wrf.getvar(ncfile, "td2",wrf.ALL_TIMES,units='K')

    atts = {"standard_name": "air_dewpoint_temperature",
            "long_name":  "Surface air dewpoint temperature",
            "units"    :  "K"                      ,
            "hgt"       :  "2 m"                    ,
            }

    ncfile.close()
    return td2,atts




def compute_PSL(filename,inputinf=None):
    """ Function to calculate PSL using wrf-python diagnostics
        It also provides variable attribute CF-Standard
        Note: this function was also coded manually for in compute_vars.py
    """



    ncfile = nc.Dataset(filename,'r')

    # Get the sea level pressure using wrf-python
    psl = wrf.getvar(ncfile, "slp",wrf.ALL_TIMES)

    # Smooth the sea level pressure since it tends to be noisy near the mountains
    smooth_psl = wrf.smooth2d(psl, 3)

    atts = {"standard_name": "air_pressure_at_mean_sea_level",
                 "long_name":      "Sea Level Pressure"      ,
                 "units"    :      "Pa"                      ,
                }

    ncfile.close()
    return smooth_psl,atts


def compute_WA(filename,inputinf=None):
    """ Function to calculate vertical wind W from WRF OUTPUTS
        It also provides variable attributes CF-Standard
    """

    ncfile = nc.Dataset(filename,'r')

    #Get vertical wind using wrf-python

    wa = wrf.getvar(ncfile, "wa",wrf.ALL_TIMES)

    atts = {"standard_name": "vertical_wind_speed",
            "long_name":  "Vertical wind speed",
            "units"    :  "m s-1"                      ,
            "hgt"       :  ""                    ,
            }

    ncfile.close()
    return wa,atts

def compute_WSFC(filename,inputinf=None):
    """ Function to calculate vertical wind W from WRF OUTPUTS at the lowest level
        It also provides variable attributes CF-Standard
    """

    ncfile = nc.Dataset(filename,'r')

    #Get vertical wind using wrf-python

    wa = wrf.getvar(ncfile, "wa",wrf.ALL_TIMES)

    if wa.ndim == 3:
        #No times
        wa = np.squeeze(wa[0,:,:])
    if wa.ndim == 4:
        #With times
        wa = np.squeeze(wa[:,0,:,:])

    atts = {"standard_name": "vertical_wind_speed",
            "long_name":  "Vertical wind speed",
            "units"    :  "m s-1"                      ,
            "hgt"       :  "lowest level"                    ,
            }

    ncfile.close()
    return wa,atts


def compute_uvmet10(filename,inputinf=None):
    """ Function to calculate 10-m windspeed rotated to Earth coordinates
        from WRF OUTPUTS
        It also provides variable attributes CF-Standard
    """

    ncfile = nc.Dataset(filename,'r')

    #Get vertical wind using wrf-python

    wa = wrf.getvar(ncfile, "uvmet10",wrf.ALL_TIMES)

    atts = {"standard_name": "wind_speed",
            "long_name":  "Earth coordinates wind speed",
            "units"    :  "m s-1"                      ,
            "hgt"       :  "10 m"                    ,
            }

    ncfile.close()
    return wa,atts

def compute_U10MET(filename,inputinf=None):
    """ Function to calculate zonal 10-m windspeed rotated to Earth coordinates
        from WRF OUTPUTS
        It also provides variable attributes CF-Standard
    """

    ncfile = nc.Dataset(filename,'r')

    #Get vertical wind using wrf-python

    ua = np.squeeze(wrf.getvar(ncfile, "uvmet10",wrf.ALL_TIMES)[0,:])

    atts = {"standard_name": "zonal_wind_speed",
            "long_name":  "Zonal earth coordinates wind speed",
            "units"    :  "m s-1"                      ,
            "hgt"       :  "10 m"                    ,
            }

    ncfile.close()
    return ua,atts

def compute_V10MET(filename,inputinf=None):
    """ Function to calculate meridional 10-m windspeed rotated to Earth coordinates
        from WRF OUTPUTS
        It also provides variable attributes CF-Standard
    """

    ncfile = nc.Dataset(filename,'r')

    #Get vertical wind using wrf-python

    va = np.squeeze(wrf.getvar(ncfile, "uvmet10",wrf.ALL_TIMES)[1,:])

    atts = {"standard_name": "meridional_wind_speed",
            "long_name":  "Meridional earth coordinates wind speed",
            "units"    :  "m s-1"                      ,
            "hgt"       :  "10 m"                    ,
            }

    ncfile.close()
    return va,atts


def compute_WDIR10(filename,inputinf=None):
    """ Function to calculate 10-m wind direction on Earth coordinates
        from WRF OUTPUTS
        It also provides variable attributes CF-Standard
    """

    ncfile = nc.Dataset(filename,'r')

    #Get wind direction (this function calculates both speed and direction)
    # We select direction only

    uvmet10_wind = np.squeeze(wrf.getvar(ncfile, "uvmet10_wspd_wdir",wrf.ALL_TIMES)[1,:])
    atts = {"standard_name": "wind_direction",
            "long_name":  "Earth coordinates wind direction",
            "units"    :  "degrees_north"                      ,
            "hgt"       :  "10 m"                    ,
            }

    ncfile.close()
    return uvmet10_wind,atts

def compute_WSPD10(filename,inputinf=None):
    """ Function to calculate 10-m wind speed on Earth coordinates
        from WRF OUTPUTS
        It also provides variable attributes CF-Standard
    """

    ncfile = nc.Dataset(filename,'r')

    #Get wind speed (this function calculates both speed and direction)
    #We select speed only

    uvmet10_wind = np.squeeze(wrf.getvar(ncfile, "uvmet10_wspd_wdir",wrf.ALL_TIMES)[0,:])

    atts = {"standard_name": "wind_speed",
            "long_name":  "Earth coordinates wind speed",
            "units"    :  "m s-1"                      ,
            "hgt"       :  "10 m"                    ,
            }

    ncfile.close()
    return uvmet10_wind,atts

def compute_CLOUDFRAC(filename,inputinf=None):
    """ Function to calculate low med and high-leve cloud fraction
        from WRF OUTPUTS
        It also provides variable attributes CF-Standard
        Method obtained from NCL using pressure levels to determine the 3 categ.
    """


    ncfile = nc.Dataset(filename,'r')

    cloudfrac = wrf.getvar(ncfile, "cloudfrac",wrf.ALL_TIMES)


    atts = {"standard_name": "cloudfrac",
            "long_name":  "Cloud fraction",
            "units"    :  "%"                      ,
            "hgt"       :  "low_mid_high",
            "low_thres": "300 m agl",
            "mid_thres": "2000 m agl",
            "high_thres": "6000 m agl",

            }

    ncfile.close()
    return cloudfrac, atts

def compute_P(filename,inputinf=None):
    """ Function to calculate pressure at Full model levels in hPa
        from WRF OUTPUTS.
        It also provides variable attributes CF-Standard
    """


    ncfile = nc.Dataset(filename,'r')

    pressure = wrf.getvar(ncfile, "pressure",wrf.ALL_TIMES)

    atts = {"standard_name": "pressure",
            "long_name":  "full_model_level_pressure",
            "units"    :  "hPa"                      ,
            "hgt"       :  ""                    ,
            }

    ncfile.close()
    return pressure, atts

def compute_GEOPOT(filename,inputinf=None):
    """ Function to calculate geopotential at Full model levels in m2 s-2
        from WRF OUTPUTS.
        It also provides variable attributes CF-Standard
    """


    ncfile = nc.Dataset(filename,'r')

    geopot = wrf.getvar(ncfile, "geopotential",wrf.ALL_TIMES)

    atts = {"standard_name": "geopotential",
            "long_name":  "full_model_level_geopotential",
            "units"    :  "m2 s-2"                      ,
            "hgt"       :  ""                    ,
            }

    ncfile.close()
    return geopot, atts

def compute_Z(filename, inputinf=None):
    """ Function to calculate height at Full model levels in m
        from WRF OUTPUTS.
        It also provides variable attributes CF-Standard
    """


    ncfile = nc.Dataset(filename,'r')

    height = wrf.getvar(ncfile, "height",wrf.ALL_TIMES)

    atts = {"standard_name": "height",
            "long_name":  "full_model_level_height",
            "units"    :  "m"                      ,
            "hgt"       :  ""                    ,
            }

    ncfile.close()
    return height, atts

def compute_TA(filename, inputinf=None):
    """ Function to calculate temperature in kelvin from WRF outputs
        It also provides variable attributes CF-Standard
    """


    ncfile = nc.Dataset(filename,'r')

    tk = wrf.getvar(ncfile,"tk",wrf.ALL_TIMES)

    atts = {"standard_name": "Temperature",
            "long_name":  "air_temperature",
            "units"    :  "K"              ,
            "hgt"       :  ""              ,
            }

    ncfile.close()

    return tk,atts

def compute_UA(filename, inputinf=None):
    """ Function to calculate earth-rotated Eastward wind components in m s-1 from WRF outputs
        It also provides variable attributes CF-Standard
    """


    ncfile = nc.Dataset(filename,'r')

    ua = np.squeeze( wrf.getvar(ncfile,"uvmet",wrf.ALL_TIMES)[0,:])

    atts = {"standard_name": "eastward_wind",
            "long_name":  "Eastward Wind",
            "units"    :  "m s-1"                      ,
            "hgt"       :  ""                    ,
            }

    ncfile.close()

    return ua,atts

def compute_VA(filename, inputinf=None):
    """ Function to calculate earth-rotated Northward wind components in m s-1 from WRF outputs
        It also provides variable attributes CF-Standard
    """


    ncfile = nc.Dataset(filename,'r')

    va = np.squeeze( wrf.getvar(ncfile,"uvmet",wrf.ALL_TIMES)[1,:])

    atts = {"standard_name": "northward_wind",
            "long_name":  "Northward Wind",
            "units"    :  "m s-1"                      ,
            "hgt"       :  ""                    ,
            }

    ncfile.close()

    return va,atts

def compute_TC(filename, inputinf=None):
    """ Function to calculate temperature in degC at model full levels from WRF outputs
        It also provides variable attributes CF-Standard
    """


    ncfile = nc.Dataset(filename,'r')

    tc =  wrf.getvar(ncfile,"tc",wrf.ALL_TIMES)

    atts = {"standard_name": "air_temperature",
            "long_name":  "Air temperature",
            "units"    :  "degC"                      ,
            "hgt"       :  "full_model_level"                    ,
            }

    ncfile.close()

    return tc,atts

def compute_TD(filename, inputinf=None):
    """ Function to calculate dewpoint temperature in degC at model full levels from WRF outputs
        It also provides variable attributes CF-Standard
    """


    ncfile = nc.Dataset(filename,'r')

    td = wrf.getvar(ncfile,"td",wrf.ALL_TIMES)

    atts = {"standard_name": "air_dewpoint_temperature",
            "long_name":  "Air Dewpoint temperature",
            "units"    :  "degC"                      ,
            "hgt"       :  "full_model_level"                    ,
            }

    ncfile.close()

    return td,atts


def compute_RH(filename, inputinf=None):
    """ Function to calculate relative humidity at model full levels from WRF outputs
        It also provides variable attributes CF-Standard
    """


    ncfile = nc.Dataset(filename,'r')

    rh = wrf.getvar(ncfile,"rh",wrf.ALL_TIMES)

    atts = {"standard_name": "relative_humidity",
            "long_name":  "Relative Humidity",
            "units"    :  "%"                      ,
            "hgt"       :  "full_model_level"                    ,
            }

    ncfile.close()

    return rh,atts

def compute_SPECHUM(filename, inputinf=None):
    """ Function to calculate sepcific humidity at model full levels from WRF outputs
        It also provides variable attributes CF-Standard
    """


    ncfile = nc.Dataset(filename,'r')

    qvapor = ncfile.variables['QVAPOR'][:]

    sh = qvapor/(1+qvapor)

    atts = {"standard_name": "specific_humidity",
            "long_name":  "Selative Humidity",
            "units"    :  "kg kg-1"                      ,
            "hgt"       :  "full_model_level"                    ,
            }

    ncfile.close()

    return sh,atts

def compute_HUSS(filename, inputinf=None):
    """ Function to calculate specific humidity near surface from WRF outputs
        It also provides variable attributes CF-Standard
    """

    ncfile = nc.Dataset(filename,'r')

    q2 = ncfile.variables['Q2'][:]

    huss = q2/(1+q2)

    atts = {"standard_name": "specific_humidity",
            "long_name":  "Surface specific humidity",
            "units"    :  "kg/kg"                      ,
            "hgt"       :  "2 m"                    ,
            }

    ncfile.close()

    return huss,atts

def compute_SST(filename, inputinf=None):
    """ Function to calculate sea surface temperature from WRF outputs
        It also provides variable attributes CF-Standard
    """
    ncfile = nc.Dataset(filename,'r')

    sst = ncfile.variables['SST'][:]
    landmask = ncfile.variables['LANDMASK'][:]

    atts = {"standard_name": "sea_surface_temperature",
            "long_name":  "Sea surface temperature",
            "units"    :  "K"                      ,
            "_FillValue": const.missingval,
            }

    sst[mask==1]=const.missingval

    ncfile.close()

    return sst,atts

def compute_OLR(filename, inputinf=None):
    """ Function to calculate Outgoing longwave radiation from WRF outputs
        It also provides variable attributes CF-Standard
    """

    ncfile = nc.Dataset(filename,'r')
    olr = ncfile.variables['OLR'][:]

    atts = {"standard_name": "outgoing_longwave_radiation",
            "long_name":  "Top-of-atmosphere outgoing longwave radiation",
            "units"    :  "W m-2"                      ,
            }

    ncfile.close()

    return olr,atts

def compute_PSFC(filename, inputinf=None):

    """ Function to calculate surface pressure from WRF OUTPUTS
        It also provides variable attributes CF-Standard
    """
    ncfile = nc.Dataset(filename,'r')

    psfc = ncfile.variables['PSFC'][:]

    atts = {"standard_name": "surface_pressure",
            "long_name":  "Surface pressure",
            "units"    :  "Pa"                      ,
            "hgt"       :  ""                    ,
            }

    ncfile.close()

    return psfc,atts



def compute_ET(filename, inputinf=None):
    """ Function to calculate surface evapotranspiration flux from WRF outputs
        It also provides variable attributes CF-Standard
    """

    ncfile = nc.Dataset(filename,'r')
    et_acc = ncfile.variables['SFCEVP'][:]

    ## Specific to accumulated variables
    if hasattr(ncfile,'PREC_ACC_DT'):
        accum_dt = getattr(ncfile,'PREC_ACC_DT')
    else:
        print(("NO PREC_ACC_DT in input file. Set to default %s min" %(inputinf['acc_dt'])))
        accum_dt = int(inputinf['acc_dt'][:])

    ## Deacumulating over prac_acc_dt (namelist entry)
    et = et_acc/(accum_dt*60.)

    atts = {"standard_name": "surface_evaporation",
            "long_name":  "surface evaporation flux",
            "units"    :  "kg m-2 s-1"                      ,
            }

    ncfile.close()

    return et,atts

def compute_CAPE2D(filename,inputinf=None):
    """ Function to calculate CAPE using methods described in:
        http://wrf-python.readthedocs.io/en/latest/user_api/generated/wrf.cape_2d.html
        This is NOT CF-compliant in any way. cape2d contains 4 variables distributed by levels: MCAPE [J kg-1], MCIN[J kg-1], LCL[m] and LFC[m]
    """

    ncfile = nc.Dataset(filename,'r')
    pres_hpa = wrf.getvar(ncfile,"pressure",wrf.ALL_TIMES)
    tkel = wrf.getvar(ncfile,"tk",wrf.ALL_TIMES)
    qv = wrf.getvar(ncfile,"QVAPOR",wrf.ALL_TIMES)
    z = wrf.getvar(ncfile,"geopotential",wrf.ALL_TIMES)/const.g
    psfc = wrf.getvar(ncfile,"PSFC",wrf.ALL_TIMES)/100. #Converto to hPA
    terrain = wrf.getvar(ncfile,"ter",wrf.ALL_TIMES)
    ter_follow=True

    cape2d = wrf.cape_2d(pres_hpa, tkel, qv, z, terrain, psfc, ter_follow,missing=const.missingval, meta=False)

    atts = {"standard_name": "cape2d_variables",
            "long_name":  "mcape mcin lcl lfc",
            "units"    :  "SI"                ,
            }

    return cape2d,atts

def compute_Q2DIV(filename,inputinf=None):
    """ Function to calculate moisture divergence """

    ncfile=nc.Dataset(filename,'r')

    u,v = wrf.getvar(ncfile,"uvmet10",wrf.ALL_TIMES)
    q2  = wrf.getvar(ncfile,"Q2",wrf.ALL_TIMES)
    dx = ncfile.DX
    dy = ncfile.DY

    Q2DIV = compute_div_dx(u*q2,v*q2,dx,dy)

    atts = {"standard_name": "moisture_divergence",
            "long_name"    : "water mixing ratio divergence",
            "units"        : "s-1",
            "hgt"       :  "2m"                    }

    return Q2DIV,atts

def compute_QDIV(filename,inputinf=None):
    """ Function to calculate moisture divergence 3D"""
    ncfile=nc.Dataset(filename,'r')
    ua = wrf.getvar(ncfile,"U",wrf.ALL_TIMES)
    va = wrf.getvar(ncfile,"V",wrf.ALL_TIMES)
    qv = wrf.getvar(ncfile,"QVAPOR",wrf.ALL_TIMES)
    dx = ncfile.DX
    dy = ncfile.DY

    QDIV = compute_div_dx(ua*qv,va*qv,dx,dy)

    atts = {"standard_name": "moisture_divergence",
            "long_name"    : "water vapor mixing ratio divergence",
            "units"        : "s-1"}

    return QDIV,atts

def compute_THETAE(filename,inputinf=None):
    """ Function to calculate equivalent potential temperature """

    ncfile=nc.Dataset(filename,'r')

    theta_e = wrf.getvar(ncfile,"theta_e",wrf.ALL_TIMES)

    atts = {"standard_name": "theta_e",
            "long_name"    : "equivalent potential temperature",
            "units"        : "K"}

    return theta_e,atts

def compute_CLDFRA(filename,inputinf=None):
    """ Function to calculate cloud fraction at mass full levels
        as estimated by WRF"""

    ncfile=nc.Dataset(filename,'r')

    cldfra = wrf.getvar(ncfile,"CLDFRA",wrf.ALL_TIMES)

    atts = {"standard_name": "Cloud_Fraction",
            "long_name"    : "cloud fraction at mass levels",
            "units"        : "m3 m-3"}

    return cldfra,atts

def compute_CTT(filename,inputinf=None):
    """Function to calculate cloud top temperature
    """

    ncfile = nc.Dataset(filename,'r')

    ctt = wrf.getvar(ncfile,'ctt',wrf.ALL_TIMES,units='K',fill_nocloud=True,missing=const.missingval)

    atts = {"standard_name": "Cloud_Top_Temperature",
            "long_name"    : "cloud top temperature",
            "units"        : "K"}

    return ctt, atts

def compute_SMOIS(filename,inputinf=None):
    """ Function to calculate soil moisture at different levels
    """

    ncfile = nc.Dataset(filename,'r')

    smois = wrf.getvar(ncfile,'SMOIS',wrf.ALL_TIMES)
    zs = wrf.getvar(ncfile,'ZS',wrf.ALL_TIMES)

    atts = {'standard_name': "Soil_Moisture",
            'long_name': "Soil moisture in model soil layers",
            'units'        : 'm3 m-3',
            'layers'       : zs}

    return smois,atts

def compute_TT(filename, inputinf=None):
    """ Function to calculate temperature in degC at model full levels from WRF outputs
        It also provides variable attributes CF-Standard
    """


    ncfile = nc.Dataset(filename,'r')
    theta = ncfile['T'][:] + 300.
    ptot  = ncfile['PB'][:] + ncfile['P'][:]
    temp = theta*(ptot/1000)*(2/7)

    atts = {"standard_name": "air_temperature",
            "long_name":  "Air temperature",
            "units"    :  "degC"                      ,
            "hgt"       :  "full_model_level"                    ,
            }

    ncfile.close()

    return temp,atts

def compute_PW(filename,inputinf=None):
    """ Function to calculate precipitable water in the column
    """

    ncfile=nc.Dataset(filename,'r')

    pw = wrf.getvar(ncfile,"pw",wrf.ALL_TIMES)

    atts = {"standard_name": "precipitable_water",
            "long_name"    : "column precipitable water",
            "units"        : "kg m-2"}

    return pw,atts

def compute_OMEGA(filename,inputinf=None):
    """ Function to calculate vertical velocity in pressure units
    """

    ncfile=nc.Dataset(filename,'r')

    omega = wrf.getvar(ncfile,"omega",wrf.ALL_TIMES)

    atts = {"standard_name": "omega",
            "long_name"    : "vertical windspeed in pressure units",
            "units"        : "Pa s-1"}

    return omega,atts

def compute_G5OMEGA(filename,inputinf=None):
    """ Function to calculate vertical velocity in pressure units
        Gaussian Filtered in the horizontal
    """

    ncfile=nc.Dataset(filename,'r')

    omega = wrf.getvar(ncfile,"omega",wrf.ALL_TIMES)
    aux = gaussian_filter(omega.values,(0,5,5),mode='nearest')

    omega.values=aux


    atts = {"standard_name": "omega",
            "long_name"    : "vertical windspeed in pressure units (gaussian filtered 5 std)",
            "units"        : "Pa s-1"}

    return omega,atts

def compute_W5L(filename,inputinf=None):
    """ Function to calculate vertical velocity in the lowest 5 levels
    """

    ncfile = nc.Dataset(filename,'r')

    w5l =  ncfile['W'][:,:5,:,:].mean(axis=1).squeeze()

    atts = {"standard_name": "W5L",
            "long_name"    : "vertical windspeed in the lowest 5 model levels",
            "units"        : "m s-1"}

    return w5l,atts


def compute_TH5L(filename,inputinf=None):
    """ Function to calculate theta in the lowest 5 levels
    """

    ncfile = nc.Dataset(filename,'r')

    th5l = ncfile['T'][:,:5,:,:].mean(axis=1).squeeze() + 300.


    atts = {"standard_name": "TH5L",
            "long_name"    : "theta in lowest 5 model levels",
            "units"        : "K"}

    return th5l,atts

def compute_THETA(filename,inputinf=None):
    """ Function to calculate theta
    """
    ncfile = nc.Dataset(filename,'r')
    th = wrf.getvar(ncfile,"theta",wrf.ALL_TIMES) #in hPA

    atts = {"standard_name": "THETA",
            "long_name"    : "potential temperature",
            "units"        : "K"}

    return th,atts

def compute_TVAVG(filename,inputinf=None):
    """ Function to calculate air density at model levels using
        ideal gas law approximation
    """

    epsilon_gamma = const.epsilon_gamma
    Rd=const.Rd# J kg-1 K-1
    #Rv=const.Rv# J kg-1 K-1
    #cp=1005 # J kg-1 K-1 at 300K
    #c_pv=1864# J kg-1 K-1 at 300K
    ncfile = nc.Dataset(filename,'r')



    p = wrf.getvar(ncfile,"pressure",wrf.ALL_TIMES) #in hPA
    #qvapor = ncfile.variables['QVAPOR'][:]
    #p_v = qvapor*p(100.*(epsilon_gamma+qvapor)) # in hPA
    tk = wrf.getvar(ncfile,"tk",wrf.ALL_TIMES)
    zstag = wrf.getvar(ncfile,"zstag",wrf.ALL_TIMES) #in m


    #p_d=p-p_v

    #rho_d=p_d*100.*tk/Rd #Converting from hPa to Pa
    #rho_v=p_v*100.*tk/Rs_v
    rho_d=p*100.*tk/Rd

    #m_d = rho_d*thick_lev
    #m_v = rho_d*thick_lev


    if zstag.ndim == 3:

        #No times
        thick_lev = zstag[1:,:,:]
        m_d = rho_d*thick_lev
        m_v = rho_d*thick_lev
        #tvavg = np.average(tk,axis=0,weights=(m_d*c_pd + m_v*c_pv)).squeeze()
        tvavg = np.average(tk,axis=0,weights=(rho_d)).squeeze()
    if zstag.ndim == 4:
        #With times
        #No times
        thick_lev = zstag[:,1:,:,:]
        m_d = rho_d*thick_lev
        m_v = rho_d*thick_lev
        tvavg = np.average(tk,axis=1,weights=(rho_d)).squeeze()
        #tvavg = np.average(tk,axis=1,weights=(m_d*c_pd + m_v*c_pv)).squeeze()


    atts = {"standard_name": "TVAVG",
            "long_name"    : "mean vertical atmospheric temperature",
            "units"        : "K"}

    return tvavg,atts

def compute_rho(filename,inputinf=None):
    """ Function to approximate air density
    """

    Rd=const.Rd

    ncfile = nc.Dataset(filename,'r')

    p = wrf.getvar(ncfile,"pressure",wrf.ALL_TIMES) #in hPA
    tk = wrf.getvar(ncfile,"tk",wrf.ALL_TIMES)

    rho=p*100.*tk/Rd



    atts = {"standard_name": "RHO",
            "long_name"    : "air density",
            "units"        : "kg m-3"}

    return rho,atts
