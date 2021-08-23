
### CMIP6_ERA5_PGW

A tool to generate Pseudo-Global Warming (PGW) boundary conditions using CMIP6 and ERA5.

Author: Daniel Argueso <daniel>
Date:   21 July 2021
Email:  d.argueso@uib.es

======

### Python environment

- Create conda CMIP6_ERA5_PGW environment: `conda env create -f cmip6era5pgw.yml`
- Activate new environment: `conda activate cmip6era5pgw`

## Requirements


## Steps to run the tool

1. Download ERA5 data using Get_ERA5_ECMWF_plevs.py and Get_ERA5_ECMWF_sfc.py scripts (you need to install and set up the cdsapi: https://cds.climate.copernicus.eu/api-how-to)
2. Convert original ERA5 in GRIB into NetCDF. Use grib2netcdf.py. Depending on the cdo version, the outputs may change (variable names, order of pressure levels). This can be adapted later on when merging ERA5 and CMIP6 data.
3. Download CMIP6 data from ESGF. Get as many models as you can. We used different methods to get a complete set (NCI scripts, ESGF wget, ESGF globus). If you have an NCI account you may follow the steps (see below) to get CMIP6 data there, otherwise they are avaiable in any of the ESGF nodes.
  - Both present (historical) and future periods (e.g. sps585) are required. Depending on the periods used to calculate the signal you may need different years. In this exampled we used 1990-2014 for present and 2076-2100 for future
  - Only monthly means are needed ("Amon")
  - Required var:
    * 3D: ta, ua, va, zg, hus
    * 2D: uas, vas, tas, ts, hurs,ps, psl

4. Once the monthly CMIP6 data is downloaded, calculate the monthly annual cycle for the periods selected (present and future), then calculate the CC signal between those two files.
    python  Calculate_CMIP6_Annual_cycle-CC_change-regrid_ERA5.py
5. Create a ERA5 grid in text file from griddes for CDO remapping (used to interpolate to a common ERA5 grid)
    cdo griddes era5_daily_sfc_[sampledate].nc > era5_grid
6. Interpolate (remap) all files to ERA5 grid:
    for file in $(ls *_AnnualCycle.nc); do cdo -remapcon,era5_grid ${file} regrid_era5/${file};done
  Note 1: this was giving an error  "Unsupported file structure" possibly because of the time dimension, or other variables not supported. The new version of the script fixes this.
  Note 2: Once the CC files are created and regridded, MCM-UA-1-0 gives some error because it has some extra variables that need to be removed
    Example:
    ncks -x -v areacella,height ts_MCM-UA-1-0_r1i1p1f2_CC_2076-2100_1990-2014_AnnualCycle.nc aux.nc
    mv aux.nc ts_MCM-UA-1-0_r1i1p1f2_CC_2076-2100_1990-2014_AnnualCycle.nc
7. Create the ensemble mean of Climate Change signal. Example:
    cdo ensmean ts_* ts_CC_signal_ssp585_2076-2100_1990-2014.nc

8. Vertically interpolate from CMIP6 pressure levels to ERA5 pressure levels.
    python Interpolate_CMIP6_Annual_cycle-CC_pinterp.py

9. Merge ERA5 and CMIP5 anomalies into single WRF-intermediate files.
  - Convert outputInter.f90 tp python module using f2py:
    f2py -c -m outputInter outputInter.f90 -DF2PY_REPORT_ON_ARRAY_COPY=1000000
  - Rename mv outputInter.[cpython-37m-x86_64-linux-gnu].so outputInter.so if necessary to outputInter.so
  - Run write_intermediate_ERA5_CMIP6anom.py which makes use of outputInter.f90 (as a python module), constanst.py, wrf_variables.py. It basically interpolates CMIP6 anomalies to every 6 hours (from monthly) and builds the WRF-Intermediate adding CMIP6 anomalies and ERA5 fields. Depending on CDO version variables in the ERA5 netCDF files may have names or codes, modify the vars2d_codes and vars3d_codes accordingly. Currently working with varcodes instead of names.


NOTE: THIS PART DIDN"T WORK WITH LATEST VERSION OF ERA5 and CDO, it has problems with depth levels
We had to create a climatology with netcdfs and directly write intemediate files for each of the starting dates
10. Create a file with soil variables to initalize. Most GCMs do not write out soil variables. We create a climatological file from ERA5 that is simply used to initialize the model. In this example we initialize our experiments in July, so we use data from June to August to create the climatology, but other dates may apply to different experiments. We no longer use a script for this like before, we do it manually.

    cdo ensmean era5_daily_sfc_20??0[6-8]??.grb SOILCLIM_June-Aug.grb
    cdo setdate,2020-07-22 SOILCLIM_June-Aug.grb aux.grb
    cdo settime,00:00:00 aux.grb SOILCLIM.20200722_June-Aug.grb

11. We move this new SOICLIM file to WPS. There we use Vtable.ERA5.SOIL1ststep (link to Vtable) and adapt the dates in the namelist_soilera5_cmip6_pgw.wps (copy to namelist.wps before). We also have to indicate constant files (see namelist_soilera5_cmip6_pgw.wps for an example). We run ./ungrib.exe to intermediate files (e.g. SOILERA5:2020-07-22_00)

INSTEAD WE DID THE FOLLOWING TO CREATE SOIL VARIABLE INTERMEDIATE FILES:

10. Steps to create soil variables climatology:
  ncea era5_daily_sfc_20??07??.nc aux.nc
  ncra aux.nc sfcclim.nc
  f2py -c -m outputInter_soil outputInter_soil.f90 -DF2PY_REPORT_ON_ARRAY_COPY=1000000 #This outputInter_soil.f90 was created from the original outputInter.f90
  mv outputInter_soil.cpython-37m-x86_64-linux-gnu.so outputInter_soil.so
  python write_intermediate_ERA5_CMIP6anom_SOILCLIM.py -s 2010 -e 2020 #Same for this one, it was created from the originla write_intermediate_ERA5_CMIP6anom.py



  [TO BE CHECKED STILL]
12. We also need to create a LSERA5 file (this was already done for regular ERA5 runs) to adequately create a landsea mask. The link the Vtable.ERA5.LANDSEA and modifiy the namelist.wps to change the output name (LSERA5) and run ungrib.exe. [This step may not be entirely necessary, but ERA-Interim had important issues with landsea mask and SST fields, they did not coincide in space. In fact, if the correct METGRID.TBL file is not used, LSERA won't be used either. If we want to use it Use METGRID.TBL.ARW_PGW, which uses the right landsea mask for SST. We didn't for present or future runs in EPICC project. ERA5 does not show large differences between default LSMASK and this version]

13. We should have two files:

    *LSERA5:2012-12-22_00 (not necessary, we use it but not in the METGRID.TBL)
    SOILERA5:2020-07-22_00

14. Then run normally. The namelist.wps should call SOILERA5 and *LSERA5 (constant fields)

15. New compilation of WRF and WPS using module_initialize_real.F_modified, which looks for soil variables only in the first timestep.
16. Run WPS and WRF normally to generate BDY conds, except that ungrib.exe is not needed, since WRF-intermediate files were already created.

## Steps to get CMIP6 from NCI
1. Log in to gadi (NCI):
    ssh -l [username] gadi.nci.org.au
2. Go to: (where your data will be copied)
    cd /g/data/[project]/[username]/CMIP6
3. Load required modules (tools to locate Data: clef and other)
    module use /g/data3/hh5/public/modules
4. Use the script to locate data (python SearchCMIPData_NCI.py), which generates list of files (text) with available models and location at NCI.
We start by running for 3D data (pressure levels) and then 2D (near-surface) but we can do it all at once. Once we have a set of files (e.g., SearchLocation_psl_cmip6_historical_mon.txt) we can run the script that finds the intersection between all variables and all experiments (hist, ssp585) to download only models available in all cases (python Get_CMIP6_Monthly_PGW_NCI.py)

NOTE: data was then rsynced from CCRC servers to UIB servers. Data was also downloaded using CMIP6 interface (wget and Globus)
