path_in = "/vg5/dargueso-NO-BKUP/postprocessed/unified/EPICC/"
patt_in = "UIB"
path_out = "/home/dargueso/Analyses/EPICC/"
path_cmaps = "/home/dargueso/share/colormaps/"
geoem_in = "/home/dargueso/share/geo_em_files/EPICC"
path_bdy = "/home/dargueso/OBS_DATA/ERA5/"
patt_bdy = "era5_monthly_prec"
bdy_data = "ERA5"
path_wrfout = "/vg6/dargueso-NO-BKUP/WRF_OUT/EPICC"
path_postproc = "/vg5/dargueso-NO-BKUP/postprocessed/EPICC/"

institution = "UIB"
patt_wrf = 'wrfout'
dom = 'd01'

syear = 2020
eyear = 2020

smonth = 8
emonth = 12

wrf_runs = ['EPICC_2km_ERA5_HVC_GWD']


reg_coords = {'BAL':[38.6,0.9,40.3,4.7],
              'CAT':[40.3,0,43,3.5]}

crosssect_coords = {'BAL':[38.8,0.95,40.2,4.4],
                    'CAT':None}



loc_coords = {'PMI': [39.56,2.74]}

region = 'EPICC'
csect  = 'BAL'
loc = 'PMI'

ref_res = '2'

wrun_ref = 'EPICC_%skm_ERA5_HVC_GWD' %(ref_res)
geofile_ref = '%s/geo_em.d01.EPICC_2km_ERA5_HVC_GWD.nc'%(geoem_in)
file_ref = 'wrfout_d01_2020-08-01_00:00:00'

zb = 5 #Buffer zone in maps

obs_ref = 'CMORPH_CRT'

zlevs=[0.05,0.1,0.2,0.5,0.75,1,1.5,2,2.5,3,3.5,4,4.5,5,6,7,8,9,10,11,12,13,14,16,18,20]
plevs=[1000,990,975,950,925,910,900,875,850,800,750,700,600,500,400,300,200,150,100]

#Post-processing info

acc_dt = 10.
vcross = 'p'

vars_post = ['PRNC']



#