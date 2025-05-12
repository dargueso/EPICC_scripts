### Steps to run WRF V4.X with ERA5

Author: Daniel Argueso <daniel>
Date:   20 Feb 2019 [edited 21 July 2021] [edited 14 December 2021]
Email:  d.argueso@uib.es

======

### Python environment

## Requirements

## Notes for UIB users only. 

A working copy of WRF V4.2.2 can be found here: /home/dargueso/WRFV4.2.2 
It was compiled using intel compilers on mc4. You need to run it on mc4.
Before you run any WRF/WPS executables, you need to load intel and netcdf modules (module load intel/19.1 netcdf-intel)

### WORKFLOW
## Steps to run WPS/WRF with ERA5

1. Download ERA5 data using Get_ERA5_ECMWF_plevs.py and Get_ERA5_ECMWF_sfc.py scripts (you need to install and set up the cdsapi: https://cds.climate.copernicus.eu/api-how-to). All required variables are detailed in these scripts, along with their codes. 


[Update] There are two possible pathways. 

# Pathway 1

2. [Optional] Create land-sea mask variable from SST field. ERA-Interim had large differences between the default landsea mask and sst variables (coast did not match at all). Thus we generated our own land sea mask from the SST fields, which are naturally masked already. This removed some artifacts that are generated during the horizontal interpolation if using the original landsea mask (during metgrid). Also, ERA5 default landsea mask is expressed as a fraction, thus it can take values from 0 to 1. This is not always interpreted correctly by WPS. It may be possible to play around with METGRID.TBL and use a correct interpolation method to avoid this issues. We decided to follow the same approach as for ERA-Interim. We create an intermediate format file called LSERA:2021-01-01_00 that is ingested by metgrid.exe. That file contains only a variable named LANDSEA, which we will use for all timesteps as a constant field. If this approach is used, we need to modify METGRID.TBL as well (METGRID.TBL.ARW_LSERA5), so we actually use the variable LANDSEA to mask relevant fields. See the appendix I for details on how the intermediate format file is generated. You can use the already generated file under sample_LSERA5_files. 

3. Ungrib ERA5 data. 
    - We need to edit namelist.wps again to set ungrib prefix back to ERA5 (which is commonly used for ERA5 data), and set the constant_name to indicate that LANDSEA is treated as constant and obtained from a different file (sample namelist provided namelist.wps_sample). 
    - Edit dates and other information according to your experiment.
    - Link Vtable.ERA5 to Vtable (to the directory where ungrib.exe lives). Vtable.ERA5 is identical to Vtable.ECMWF provided by WRF, except for LANDSEA, which is removed. 
    - Link the ERA5 grib data files using ./link_grib.csh {path_to_ERA5_data}/era5_daily*.grb [adjust depending on your experiment]
    - Run ungrib.exe (You may run into a segmentation fault. Try: ulimit -s unlimited)

4. Generate your own domains (geogrid.exe). This can be done anytime before and it is independent from previous steps. This is specific to each simulation. Sample namelists provided here are for runs over the Western Mediterranean (EPICC runs). You need to edit them.

5. Create metgrid files (metgrid.exe). This is specific to each simulation too - you must edit accordingly. Sample namelist.wps is for EPICC runs.

    - Link the correct METGRID.TBL file if your using the newly created LSERA5 file (METGRID.TBL.ARW_LSERA5). metgrid.exe needs the METGRID.TBL file to be in metgrid folder and its name must be METGRID.TBL (use symlinks. In metgrid folder: ln -sf {path_to_metgrid_tbl_lsera5}/METGRID.TBL.ARW_LSERA5 METGRID.TBL)
    - Run metgrid.exe
    - You should get a series of met_em.d0? files (4 in the sample provided, from 2021-01-01_00 to 2021-01-01_18). Each of them should have two land-sea mask variables: LANDMASK (from WRF, hi-res) and LANDSEA (from ERA5 SST, ERA5 resolution). SST, soil variables and other land/ocean only variables are masked using LANDSEA.

# Pathway 2

[Update] In recent versions (V4.2.2), in the Mediterranean, this approach did not make a substantial difference in SST. This is likely due to the fact that the fractional land-sea mask is interpreted correctly, but we have not investigated further. We can use the default approach, but we need to provide the landmask from ERA5 (code 174). SST looks the same in our case, SOIL variables are different. It is not clear whic is the best option for soil variables. The workflow below is using the default land-sea mask from ERA5.

3. Ungrib ERA5 data. 
    - Edit namelist.wps following the sample (namelist.wps_sample_default_approach). 
    - Edit dates and other information according to your experiment.
    - Link Vtable.ECMWF to Vtable
    - Link the ERA5 grib data files using ./link_grib.csh {path_to_ERA5_data}/era5_daily*.grb [adjust depending on your experiment]
    - Run ungrib.exe (You may run into a segmentation fault. Try: ulimit -s unlimited

4. Generate your own domains (geogrid.exe). This can be done anytime before and it is independent from previous steps. This is specific to each simulation. Sample namelists provided here are for runs over the Western Mediterranean (EPICC runs). You need to edit them.

5. Create metgrid files (metgrid.exe). This is specific to each simulation too - you must edit accordingly. Sample namelist.wps_sample_default_approach is for EPICC runs.
   - Link the correct METGRID.TBL (METGRID.TBL.ARW). metgrid.exe needs the METGRID.TBL file to be in metgrid folder and its name must be METGRID.TBL (use symlinks. In metgrid folder: ln -sf {path_to_metgrid_tbl_lsera5}/METGRID.TBL.ARW METGRID.TBL). land-sea mask from ERA5 must be provided in era5 grib files! 
   - Run metgrid.exe
   - You should get a series of met_em.d0? files (4 in the sample provided, from 2021-01-01_00 to 2021-01-01_18). Each of them should STILL (as compared to the method above) have two land-sea mask variables: LANDMASK (from WRF, hi-res) and LANDSEA (from ERA5, ERA5 resolution). LANDSEA is different from LANDSEA obtained with the SST method above. SST, soil variables and other land/ocean only variables are masked using LANDMASK.


WPS is completed succesfully. Check SST and soil variables in your met_em files to see if they look fine.

*Note*: Depending on the region, you must be careful with lakes. The default is to treat them as water points, so they are assigned SST from the nearest ocean points. In most cases, this is not desirable, and you must use a lake model (sf_lake_physics = 1), and possibly make other considerations.



*Note*: A script that automatizes everything is also provided. It runs WPS and real.exe recursively. wrf.exe can also be added, although wrf.exe is usually run in supercomputers using different parameters. In fact, the EPICC runs did not use exactly these parameters and were optimized for the supercomputer. It is only a sample for western mediterranean experiments.

## Appendix I

Generation of LSERA5:2021-01-01_00 from SST field. This step is only required once. The file generated can be used for any simulation driven by ERA5.

1. We get a sample surface file from ERA5 (e.g. sample_ERA5_files/era5_daily_sfc_20210101.grb)
2. We select the first time step only: cdo seltimestep,1 era5_daily_sfc_20210101.grb aux.grb
3. We select SST variable only: cdo selcode,34 aux.grb aux2.grb
4. We set all missing values to a constant value 1 (LAND): cdo setmisstoc,1 aux2.grb aux3.grb
5. We set all values from a range 2-500 to missing :cdo setrtomiss,2,500 aux3.grb aux4.grb
6. Now we set all missing values (ocean) to a constant value 0: cdo setmisstoc,0 aux4.grb aux5.grb
7. We change the grib code of the variable from SST (34) to landsea mask (172), and generate the final grib file: cdo chcode,34,172 aux5.grb LSERA5_2012-12-22_00.grb
8. Delete all aux*.grb files

Move to your WPS folder. 

1. Link the newly created grib file (LSERA5_2012-12-22_00.grb) using ./link_grib.csh {path_to_LSERA5_file}/LSERA5_2012-12-22_00.grb 
2. Edit the namelis.wps (sample provided namelist.wps_sample_LSERA5) and set the following:
    &share
        wrf_core = 'ARW',
        max_dom = 1,
        start_date = '2021-01-01_00:00:00',
        end_date   = '2021-01-01_00:00:00',
        interval_seconds = 21600
        io_form_geogrid = 2,
    /
    
    
    &ungrib
        out_format = 'WPS',
        prefix = 'LSERA5',
    /
3. Link the Vtable.ERA5.LANDSEA to Vtable (to the directory where ungrib.exe lives)
4. Run ungrib.exe. This will generate an intermediate format file called: LSERA5:2021-01-01_00    
5. Running util/rd_intermediate.exe LSERA5\:2021-01-01_00 should give you the following:

    ================================================
    FIELD = LANDSEA
    UNITS = 0/1 Flag DESCRIPTION = Land/Sea flag
    DATE = 2021-01-01_00:00:00 FCST = 0.000000
    SOURCE = ECMWF
    LEVEL = 200100.000000
    I,J DIMS = 1200, 601
    IPROJ = 0  PROJECTION = LAT LON
    REF_X, REF_Y = 1.000000, 1.000000
    REF_LAT, REF_LON = 90.000008, 0.000000
    DLAT, DLON = -0.300000, 0.300000
    EARTH_RADIUS = 6367.470215
    DATA(1,1)=0.000000



