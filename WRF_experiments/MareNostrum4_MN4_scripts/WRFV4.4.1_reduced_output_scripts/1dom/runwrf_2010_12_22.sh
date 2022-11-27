#!/bin/bash

# @Author: Daniel Argueso <daniel>
# @Date:   2017-10-30T17:40:49+01:00
# @Email:  d.argueso@uib.es
# @Last modified by:   daniel
# @Last modified time: 2021-03-19T09:28:25+01:00

# create 3 slurm scripts to run a WRF job
# including getting the boundary data and sending the output data
# and calling the next job

#write the namelist.input
cat << EOF_namelist > ./namelist.input

&time_control
run_days                            = 0,
run_hours                           = 0,
run_minutes                         = 0,
run_seconds                         = 0,
start_year                          = 2010,
start_month                         = 12,
start_day                           = 22,
start_hour                          = 00,
start_minute                        = 00,
start_second                        = 00,
end_year                            = 2010,
end_month                           = 12,
end_day                             = 22,
end_hour                            = 06,
end_minute                          = 00,
end_second                          = 00,
interval_seconds                    = 21600,
input_from_file                     = .true.,
history_interval                    =  60,
frames_per_outfile                  =  1,
restart                             = .false.,
restart_interval                    = 1440,
override_restart_timers             = .true.,
write_hist_at_0h_rst                = .true.,
io_form_history                     = 102 
io_form_restart                     = 102
io_form_input                       = 2
io_form_boundary                    = 2
debug_level                         = 0
auxhist7_outname                    = "wrfprec_d<domain>_<date>"
auxhist7_interval                   = 10,
frames_per_auxhist7                 = 144,
io_form_auxhist7                    = 102
auxhist8_outname                    = "wrf3hrly_d<domain>_<date>"
auxhist8_interval                   = 180,
frames_per_auxhist8                 = 1,
io_form_auxhist8                    = 102,
auxinput4_inname                    = "wrflowinp_d<domain>"
auxinput4_interval                  = 360,
io_form_auxinput4                   = 2,
iofields_filename                   = "myoutfields_4.4.1.txt",
ignore_iofields_warning             = .true.,
nocolons 			    = .true.
/


&domains
time_step                           = 12,
time_step_fract_num                 = 0,
time_step_fract_den                 = 1,
max_dom                             = 1,
e_we                                = 1250,
e_sn                                = 750,
e_vert                              = 50,
p_top_requested                     = 5000,
num_metgrid_levels                  = 38
num_metgrid_soil_levels             = 4
dx                                  = 2000,
dy                                  = 2000,
grid_id                             = 1,
parent_id                           = 0,
i_parent_start                      = 1,
j_parent_start                      = 1,
parent_grid_ratio                   = 1,
parent_time_step_ratio              = 1,
feedback                            = 0,
smooth_option                       = 0,
numtiles                            = 4,
nproc_x                             = 12,
nproc_y                             = 48,
/

&physics
mp_physics                          = 6,
ra_lw_physics                       = 4,
ra_sw_physics                       = 4,
radt                                = 10,
sf_sfclay_physics                   = 1,
sf_surface_physics                  = 2,
bl_pbl_physics                      = 1,
sf_urban_physics                    = 0,
sf_lake_physics                     = 1,
bldt                                = 0,
topo_wind                           = 1,
cu_physics                          = 0,
cudt                                = 0,
shcu_physics                        = 0,
isfflx                              = 1,
ifsnow                              = 0,
icloud                              = 1,
surface_input_source                = 1,
num_soil_layers                     = 4,
num_land_cat                        = 21,
sst_update                          = 1,
tmn_update                          = 1,
lagday                              = 150,
sst_skin                            = 1,
usemonalb                           = .true.,
rdmaxalb                            =.true.,
rdlai2d                             =.true.,
slope_rad                           = 1,
topo_shading                        = 1,
shadlen                             = 25000.,
prec_acc_dt                         = 10.,
bucket_mm                           = 1000.,
bucket_J                            = 1.e9,
do_radar_ref                        = 1,
/

&fdda
grid_fdda                           = 0,

/


&dynamics
rk_ord                              = 3,
w_damping                           = 0,
diff_opt                            = 1,
km_opt                              = 4,
diff_6th_opt                        = 2,
diff_6th_factor                     = 0.12,
base_temp                           = 290.,
damp_opt                            = 3,
zdamp                               = 5000.,
dampcoef                            = 0.2,
khdif                               = 0,
kvdif                               = 0,
epssm                               = 0.3,
non_hydrostatic                     = .true.,
moist_adv_opt                       = 1,
scalar_adv_opt                      = 1,
gwd_opt                             = 0,
/

&bdy_control
spec_bdy_width                      = 5,
spec_zone                           = 1,
relax_zone                          = 4,
spec_exp                            = 0.33,
specified                           = .true.,
nested                              = .false.,
/

&grib2
/

&namelist_quilt
nio_tasks_per_group = 0,
nio_groups = 0,
/
EOF_namelist

#Create SLURM task (script)

cat << EOF_WRF > ./E220101222.slurm
#!/bin/bash
#SBATCH --job-name=E220101222
#SBATCH --output=WRF20101222-%j.out
#SBATCH --error=WRF20101222-%j.err
#SBATCH --time=16:00:00
#SBATCH --ntasks=576

module load netcdf/4.4.1.1
module load hdf5/1.8.19

module load impi/2018.1
module load mkl/2019.4

ulimit -S -s unlimited
# run the program
srun ./wrf.exe >& wrf.log

EOF_WRF

cat << \EOF_po > ./po20101222.slurm
#!/bin/bash
#SBATCH --job-name=po420101222
#SBATCH --output=po20101222-%j.out
#SBATCH --error=po20101222-%j.err
#SBATCH --time=01:00:00 # walltime, abbreviated by -t
#SBATCH --tasks=1

#sh ./runwrf_2011_01_01.sh

module load netcdf/4.4.1.1
module load hdf5/1.8.19


tempdir=wrftemp20101222

mkdir ${tempdir}

#rm ./wrfprec_d01_2011-01-01*
mv ./wrfout* $tempdir
mv ./wrfprec* $tempdir
mv ./wrf3hrly* $tempdir
mv $tempdir/* /gpfs/projects/uib30/WRF_OUT/EPICC/EPICC_2km_ERA5/out/



rmdir $tempdir

#sh ./runwrf_2011_01_01.sh

#moving restart files to project folder - if any of them fail then fail out of job
mv -v ./wrfrst_d01_2011-01-01* /gpfs/projects/uib30/WRF_OUT/EPICC/EPICC_2km_ERA5/restart/
if [ $? != 0 ]; then
    exit $?
fi

#Linking restart files needed for next run
#rm -f wrfrst_d01_2010-12-22*
#ln -sf /gpfs/projects/uib30/WRF_OUT/EPICC/EPICC_2km_ERA5/restart/wrfrst_d01_2011-01-01* .

EOF_po

#rm -fr ./wrf*d01
ln -sf /gpfs/projects/uib30/WRF_BDY/EPICC_2km_ERA5/wrfinput_d01_2010-12-22 wrfinput_d01
ln -sf /gpfs/projects/uib30/WRF_BDY/EPICC_2km_ERA5/wrflowinp_d01_2010-12-22 wrflowinp_d01
ln -sf /gpfs/projects/uib30/WRF_BDY/EPICC_2km_ERA5/wrfbdy_d01_2010-12-22 wrfbdy_d01

mjid=$(sbatch  ./E220101222.slurm | cut -f 4 -d' ')
sbatch --depend=afterok:${mjid} ./po20101222.slurm
