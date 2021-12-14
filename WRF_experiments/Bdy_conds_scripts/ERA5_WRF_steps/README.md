### Steps to run WRF V4.2.2 with ERA5

Author: Daniel Argueso <daniel>
Date:   21 July 2021
Email:  d.argueso@uib.es

======

### Python environment

## Requirements

## Steps to run the tool

1. Download ERA5 data using Get_ERA5_ECMWF_plevs.py and Get_ERA5_ECMWF_sfc.py scripts (you need to install and set up the cdsapi: https://cds.climate.copernicus.eu/api-how-to). All required variables are detailed in these scripts, along with their codes. 

2. [Optional] We need to generate a LSERA5 file to create an adequate landsea mask (LSERA5:2012-12-22_00). The link the Vtable.ERA5.LANDSEA and modifiy the namelist.wps to change the output name (LSERA5) and run ungrib.exe. This step is not necessary. ERA5 does dos not show large differences between default LSMASK and the newly generated version from SST fields. ERA-Interim did have large differences between the default LANDSEA MASK and SST variables (coasts did not match). In order to use this newly created landsea mask, METGRID.TBL also needs to be changed (METGRID.TBL.ARW_LSERA5), so the variable LANDSEA is used to mask relevant fields. 

3. Ungrib ERA5 using Vtable.ERA5 (which is identical to Vtable.ECMWF provided by WRF).

4. Generate your own domains (geogrid.exe). This can be done anytime before and it is independent from previous steps.

5. Run metgrid.exe. Use METGRID.TBL.ARW (or METGRID.TBL.ARW_LSERA5 if using LSERA5 file)