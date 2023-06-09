# Steps to run WRF V4.2.2 with ERA5 and with ERA5+CMIP6anom (PGW)
## EPICC project runs

Author: Daniel Argueso <daniel>  
Date:   9 June 2023  
Email:  d.argueso@uib.es

---

## Python environment

### Requirements

The python environment, the requirements and the package versions to run this tool are specified in the file `m4_epicc_env.yml`

---

## ERA5
### Steps to run the tool

1. Download ERA5 data using `Get_ERA5_ECMWF_plevs.py` and `Get_ERA5_ECMWF_sfc.py` scripts (you need to install and set up the cdsapi: https://cds.climate.copernicus.eu/api-how-to). All required variables are detailed in these scripts, along with their codes. 

2. Ungrib ERA5 using `Vtable.ERA5` (which is identical to `Vtable.ECMWF` provided by WRF).

3. **[Optional]** ERA-Interim had large differences between the default LANDSEA MASK and SST variables (coasts did not match). Thus we generated our own land sea mask from the SST fields which were already masked. This removed many artifacts that were generated otherwise. We followed a similar approach for ERA5 and created `LSERA5:2012-12-22_00` (intermediate format) with this method. This is not necessary in our region since ERA5 does not show large differences between default LSMASK and the newly generated version from SST fields, but it may be done in other regions. If this approach is finally used, it is necessary to change `METGRID.TBL` as well (`METGRID.TBL.ARW_LSERA5`), so the variable LANDSEA is used to mask relevant fields. **Note**: In EPICC runs, we created LANDSEA mask, but all variables were masked using LANDMASK during metgrid.exe (we used `METGRID.TBL.ARW`). No artifacts were found. To keep consistency with PGW runs (next section), it may be better to use LANDSEA in both experiments. 

4. Generate your own domains (geogrid.exe). This can be done anytime before and it is independent from previous steps.

5. Run metgrid.exe. Use METGRID.TBL.ARW (or METGRID.TBL.ARW_LSERA5 if using LSERA5 file)

Steps 2-5 can be done automatically running with the right options inside (which WPS/WRF modules to run, paths, and dates, among other):

    myrunWPSandreal_UIB_daily_EPICC_2km_ERA5.py

---

## ERA5+CMIP6anom (PGW)
### Steps to run the tool


