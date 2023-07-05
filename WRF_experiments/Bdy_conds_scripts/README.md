# Steps to run WRF V4.2.2 with ERA5 and with ERA5+CMIP6anom (PGW)
## EPICC project runs

Author: Daniel Argueso <daniel>  
Date:   9 June 2023  
Email:  d.argueso@uib.es

---

## Python environment

### Requirements

The python environment, the requirements and the package versions to run this tool are specified in the file `m4_epicc_env.yml`

- Create conda CMIP6_ERA5_PGW environment: `conda env create -f cmip6era5pgw.yml`
- Activate new environment: `conda activate cmip6era5pgw`

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

### Appendix I

Generation of LSERA5:2021-01-01_00 from SST field.

1. We get a sample surface file from ERA5 (e.g. era5_daily_sfc_20210101.grb)
2. We select the first time step only: 
3. We select SST variable:
4. We set all valid values to 0 (OCEAN)
5. We set missing value to 1 (LAND)
6. We set all values between 1 and 500 to missing (LAND)
7. We set missing to constant 1 (LAND)

```
cdo seltimestep,1 era5_daily_sfc_20210101.grb LSERA.grb  
cdo selcode,34 LSERA.grb LSERA2.grb  
cdo setmisstoc,1 LSERA2.grb LSERA4.grb  
cdo setrtomiss,2,500 LSERA4.grb LSERA5.grb  
cdo setmisstoc,0 LSERA5.grb LSERA6.grb  
cdo chcode,34,172 LSERA6.grb LSERA5_2012-12-22_00.grb
```

Then we need to convert this grb file to intermediate file using `./ungrib.exe` and `Vtable.ERA5.LANDSEA` (almost any namelist.wps with the correct dates will work)

---

## ERA5+CMIP6anom (PGW)
### Steps to run the tool

1. [if not done yet] Download ERA5 data using `Get_ERA5_ECMWF_plevs.py` and `Get_ERA5_ECMWF_sfc.py` scripts (you need to install and set up the cdsapi: https://cds.climate.copernicus.eu/api-how-to). All required variables are detailed in these scripts, along with their codes. 

2. Convert original ERA5 in GRIB into NetCDF. Use grib2netcdf.py. Depending on the cdo version, the outputs may change (variable names, order of pressure levels). This can be adapted later on when merging ERA5 and CMIP6 data.

3. Download CMIP6 data from ESGF. Get as many models as you can. You need both present (historical) and future periods (e.g. ssp585). Depending on the periods used to calcualte the signal you will need different years. In this example we used 1985-2014 for present and 2070-2099 for future
    - Only monthly means are needed ("Amon")
    - Required variables:
        * 3D: ta, ua, va zg, hur (hus is also possible but the code must be adapted)
        * 2D: uas, vas, tas, ts, hurs, ps, psl

4. Once the monthly CMIP6 data is downloaded, they must be organized. First create a list of the available models. It can be created simply using a list command. We create something like `list_CMIP6.txt `.


```
reorganize_CMIP6_folders.py
```

**Note**: You can reorganized individual variables or models too by changing the correspoding options.

5. Now you check that all the models are complete, so you can continue with the generation of the climate change signals. This also requires a list of models than can be provided inline as options or using a text file `list_CMIP6.txt`

```
check_completeness_CMIP6_PGW.py
```

6. Once the monthly CMIP6 data is downloaded, organized and complete, we calculate the monthly annual cycle for the periods selected (present and future). Then we calculate the CC signal between those two files and regrid the resulting files on the ERA5 grid (which needs to be created beforehand)

```
cdo griddes era5_daily_sfc_[sampledate].nc > era5_grid
python  Calculate_CMIP6_Annual_cycle-CC_change-regrid_ERA5.py
```

**Note 1**: This process was giving an error. "Unsupported file structure" possibly because of the time dimension, or other variables not supported. The current version of script fixes this.
**Note 2**: Once the CC files are created and regridded, MCM-UA-1-0 was givinb some error because it has some extra variables that need to be removed. The current version of the script fixes this too, but it can be manually fixed:
```
for file in $(ls *_MCM-UA*.nc); do ncks -x -v areacella,height ${file} aux.nc; mv aux.nc ${file}; done
```

7. Calculate ehte ensemble using the script `Create_CMIP6_AnnualCycleChange_ENSMEAN.py`. It also needs a list of models `list_CMIP6.txt` or provide them as input arguments. If no variables is selected in the options, all variables are processed. 

**Note 1**: This script gives an error if files for ssp585 include years past 2100. We don't know yet the source of the problem, but the message is related to the monotonic nature of the time dimension. Removing those files and keeping only files up to 2100 fixes the problem.
**Note 2**: We originally used `cdo ensmean tas_* tas_CC_signal_ssp585_2070-2099_1985-2014.nc` to create the ensemble means, but newer versions of CDO and latest data processing resulted in issues with the number of variables across files. So we decided to use the script above.

8. Files corresponding to variables in the vertical need to be interpolated from CMIP6 pressure levels to ERA5 pressure levels:

```
python Interpolate_CMIP6_Annual_cycle-CC_pinterp.py
```

9. Merge ERA5 and CMIP5 anomalies into single WRF-intermediate files.

    - Convert outputInter.f90 tp python module using f2py:
    f2py -c -m outputInter outputInter.f90 -DF2PY_REPORT_ON_ARRAY_COPY=1000000
    - Rename mv outputInter.[cpython-37m-x86_64-linux-gnu].so outputInter.so if necessary to outputInter.so
    - Run write_intermediate_ERA5_CMIP6anom.py which makes use of outputInter.f90 (as a python module), constanst.py, wrf_variables.py. It basically interpolates CMIP6 anomalies to every 6 hours (from monthly) and builds the WRF-Intermediate adding CMIP6 anomalies and ERA5 fields. Depending on CDO version variables in the ERA5 netCDF files may have names or codes, modify the vars2d_codes and vars3d_codes accordingly.

10. Create a file with soil variables to initalize. Most GCMs do not write out soil variables. There are two options here: a) We create a climatological file from ERA5 that is simply used to initialize the model or b) We get the era5 data for the day we initialize the model so we keep consistency with the present run (but information for a single day is provided instead of a climatology).

    #### a) CREATE A CLIMATOLOGY
    - Create a climatology:
        ```
        ncea era5_daily_sfc_20??07??.nc aux.nc
        ```
    - Create a python module from FROTRAN to create intermediate files from the climatology. This `outputInter_soil.f90` was created from the original `outputInter.f90`. The `write_intermediate_ERA5_CMIP6anom_SOILCLIM.py` file was also created from `write_intermediate_ERA5_CMIP6anom.py`. [In mc4 at UIB, the corresponding modules must be loaded before running this: intel/19.1 and netcdf-intel]
        ```
        f2py -c -m outputInter_soil outputInter_soil.f90 -DF2PY_REPORT_ON_ARRAY_COPY=1000000
        mv outputInter_soil.cpython-37m-x86_64-linux-gnu.so outputInter_soil.so
        python write_intermediate_ERA5_CMIP6anom_SOILCLIM.py -s 2010 -e 2020
        ```
    <!-- [TO BE CHECKED STILL, BECAUSE SOIL FILES CONTAIN LANDSEA]
    - We also need to create a LSERA5 file to correctly interpolate certain fields (e.g. SST). This was already done for regular ERA5 runs (see Apendix I above). If we want to use LANDSEA to mask fields we need to use `METGRID.TBL.ARW_PGW`, which uses that landmask (LANDSEA) for some fields.  -->

    - We should have a files with SOIL variables and LANDSEA. For example `SOILERA5:2020-12-22_00`.

    #### b) USE DAILY DATA FROM ERA5


    - In order to keep consistency with the soil variables used in present climate runs and given that most GCMs do not provide soil variables, we use the data corresponding to the starting date from ERA5. So, if we start on 2020-12-22_00, we use data from that day. We need `Vtable.ERA5.SOIL1ststep`, `myrunWPSandreal_UIB_daily_EPICC_2km_ERA5_CMIP6anom_SOILERA.py` and `namelist_wps_EPICC_2km_ERA5_CMIP6anom_SOILERA.deck`

        ```
        myrunWPSandreal_UIB_daily_EPICC_2km_ERA5_CMIP6anom_SOILERA.py
        ```

11. New compilation of WRF and WPS using module_initialize_real.F_modified, which looks for soil variables only in the first timestep.

12. Once we've created SOILERA5 files, once for every time the model is initialized (cold start). Then run normally using `METGRID.TBL.ARW_PGW` to interpolate using LANDSEA. Ideally, both present and future runs should use the same interpolation method, but LANDSEA is only used in future runs due to the coarse resolution of PGW original data.
```
python myrunWPSandreal_UIB_daily_EPICC_2km_ERA5_CMIP6anom.py 
```

This will run WPS and WRF normally to generate BDY conds, except that ungrib.exe is not needed, since WRF-intermediate files were already created.