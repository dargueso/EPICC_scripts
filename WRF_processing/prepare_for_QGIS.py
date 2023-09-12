import netCDF4 as nc
import xarray as xr
import pandas as pd
import numpy as np

fin = xr.open_dataset("UIB_MON_TAS_2011-2020.nc")

newfin = fin.rename_dims({'y':'south_north','x':'west_east'})



fgeo = xr.open_dataset("geo_em.d01.EPICC_2km_ERA5_HVC_GWD.nc")
fwprec = xr.open_dataset("wrfprec_d01_2018-08-20_00:00:00")

newfin['XLAT']=fgeo['XLAT_M'][0,:]
newfin['XLAT']=newfin['XLAT'].expand_dims(time=newfin.time)

newfin['XLONG']=fgeo['XLONG_M'][0,:]
newfin['XLONG']=newfin['XLONG'].expand_dims(time=newfin.time)


newfin['XLAT_V']=fgeo['XLAT_V'][0,:]
newfin['XLAT_V']=newfin['XLAT_V'].expand_dims(time=newfin.time)

newfin['XLONG_V']=fgeo['XLONG_V'][0,:]
newfin['XLONG_V']=newfin['XLONG_V'].expand_dims(time=newfin.time)

newfin['XLAT_U']=fgeo['XLAT_U'][0,:]
newfin['XLAT_U']=newfin['XLAT_U'].expand_dims(time=newfin.time)

newfin['XLONG_U']=fgeo['XLONG_U'][0,:]
newfin['XLONG_U']=newfin['XLONG_U'].expand_dims(time=newfin.time)

for gatt in fgeo.attrs:

    newfin.attrs[gatt] = fgeo.attrs[gatt]


newfin = newfin.drop_vars('time_bnds')
newfin = newfin.rename({'time':'Time'})

Timestr = [pd.to_datetime(str(date.values)).strftime('%Y-%m-%d_%H:%M:%S') for date in newfin['Time']]

tchar = np.zeros((len(Timestr),19),dtype='S1')
for i in range(len(Timestr)):
    tchar[i,:]= np.array(list(Timestr[i]))
                         
#newfin = newfin.expand_dims(dim={"DateStrLen": 19})
newfin = newfin.drop_indexes('Time')

ds = xr.Dataset({'Times': (('Time', 'DateStrLen'), tchar)})

newfin['Times'] = ds['Times']

newfin = newfin.drop_vars(["Time","lon","lat"])
# newfin = newfin.rename({'RAIN':'PREC_ACC_NC'})

# for varname in newfin.variables:


#     newfin[varname].attrs=[]

#     for vatt in fwprec[varname].attrs:
#         newfin[varname].attrs[vatt] = fwprec[varname].attrs[vatt]

newfin.to_netcdf("aux_gis.nc",unlimited_dims='Time')

#Need to install NCO
#ncwa -a string1 aux_gis.nc aux_gis2.nc
