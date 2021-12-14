### Steps to run WRF V4.2.2 with ERA5

Author: Daniel Argueso <daniel>
Date:   21 July 2021
Email:  d.argueso@uib.es

======

### Python environment

## Requirements

## Steps to run the tool

1. Download ERA5 data using Get_ERA5_ECMWF_plevs.py and Get_ERA5_ECMWF_sfc.py scripts (you need to install and set up the cdsapi: https://cds.climate.copernicus.eu/api-how-to). All required variables are detailed in these scripts, along with their codes. 

2. [Optional] ERA-Interim had large differences between the default LANDSEA MASK and SST variables (coasts did not match). Thus we generated our own land sea mask from the SST fields which were already masked. This removed many artifacts that were generated otherwise. We followed a similar approach for ERA5 and created LSERA5:2012-12-22_00 (intermediate format) with this method. This is not necessary in our region since ERA5 does not show large differences between default LSMASK and the newly generated version from SST fields, but it may be done in other regions. If this approach is finally used, it is necessary to change METGRID.TBL as well (METGRID.TBL.ARW_LSERA5), so the variable LANDSEA is used to mask relevant fields. 

3. Ungrib ERA5 using Vtable.ERA5 (which is identical to Vtable.ECMWF provided by WRF).

4. Generate your own domains (geogrid.exe). This can be done anytime before and it is independent from previous steps.

5. Run metgrid.exe. Use METGRID.TBL.ARW (or METGRID.TBL.ARW_LSERA5 if using LSERA5 file)


Note: I also attached a couple of scripts that automatize everything. One is for running WPS recursively. Another one is for running real.exe.

## Appendix I

Generation of LSERA5:2021-01-01_00 from SST field.

1. We get a sample surface file from ERA5 (e.g. era5_daily_sfc_20210101.grb)
2. We select the first time step only: 
3. We select SST variable:
4. We set all valid values to 0 (OCEAN)
5. We set missing value to 1 (LAND)
6. We set all values between 1 and 500 to missing (LAND)
7. We set missing to constant 1 (LAND)

cdo seltimestep,1 era5_daily_sfc_20210101.grb LSERA.grb
cdo selcode,34 LSERA.grb LSERA2.grb
cdo setmisstoc,1 LSERA2.grb LSERA4.grb
cdo setrtomiss,2,500 LSERA4.grb LSERA5.grb
cdo setmisstoc,0 LSERA5.grb LSERA6.grb
cdo chcode,34,172 LSERA6.grb LSERA5_2012-12-22_00.grb

