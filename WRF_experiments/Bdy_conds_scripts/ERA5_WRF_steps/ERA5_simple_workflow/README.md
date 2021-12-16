### Steps to run WRF V4.2.2 with ERA5

Author: Daniel Argueso <daniel>
Date:   20 Feb 2019 [edited 21 July 2021] [edited 14 December 2021]
Email:  d.argueso@uib.es

Edited from steps to run EPICC simulations for Angkur Sati.
This is the basic workflow only - other pathways are also possible with different treatment of landmask, sst and skintemp.
======

### WORKFLOW
## Steps to run WPS/WRF with ERA5

1. Download ERA5 data using Get_ERA5_ECMWF_plevs.py and Get_ERA5_ECMWF_sfc.py scripts (you need to install and set up the cdsapi: https://cds.climate.copernicus.eu/api-how-to). All required variables are detailed in these scripts, along with their codes. 

2. Ungrib ERA5 data. 
    - Edit namelist.wps following the sample (namelist.wps_sample). 
    - Edit dates and other information according to your experiment.
    - Link Vtable.ECMWF to Vtable
    - Link the ERA5 grib data files using ./link_grib.csh {path_to_ERA5_data}/era5_daily*.grb [adjust depending on your experiment]
    - Run ungrib.exe (You may run into a segmentation fault. Try: ulimit -s unlimited)

3. Generate your own domains (geogrid.exe). This can be done anytime before and it is independent from previous steps. This is specific to each simulation. Sample namelists provided here are for test experiments over Ireland. You need to edit them.

4. Create metgrid files (metgrid.exe). This is specific to each simulation too - you must edit accordingly. Sample namelist.wps_sample_default_approach is for EPICC runs.
   - Link the correct METGRID.TBL (METGRID.TBL.ARW). metgrid.exe needs the METGRID.TBL file to be in metgrid folder and its name must be METGRID.TBL (use symlinks. In metgrid folder: ln -sf {path_to_metgrid_tbl_lsera5}/METGRID.TBL.ARW METGRID.TBL). land-sea mask from ERA5 must be provided in era5 grib files! 
   - Run metgrid.exe
   - You should get a series of met_em.d0? files (4 in the sample provided, from 2021-01-01_00 to 2021-01-01_18). Each of them should have two land-sea mask variables: LANDMASK (from WRF, hi-res) and LANDSEA (from ERA5, ERA5 resolution).  Soil variables and other land/ocean only variables are masked using LANDMASK.

WPS is completed succesfully. Check SST and soil variables in your met_em files to see if they look fine.

5. Create boundary conditions. 
    - Use namelist.input as an example to run real.exe. It can be used to run wrf.exe too, but this configuration has not been tested. This namelist is inherited from previous runs.
    - You may run into a segmentation fault. Try: ulimit -s unlimited (if you haven't done so for ungrib.exe)

*Note*: Depending on the region, you must be careful with lakes. The default is to treat them as water points, so they are assigned SST from the nearest ocean points. In most cases, this is not desirable, and you must use a lake model (sf_lake_physics = 1), and possibly make other considerations.
