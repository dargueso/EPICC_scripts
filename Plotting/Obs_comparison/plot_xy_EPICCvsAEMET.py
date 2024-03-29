from os import sep
import pdb
import pandas as pd
import netCDF4 as nc
import xarray as xr
import numpy as np
from glob import glob

import epicc_config as cfg
from wrf import ll_to_xy

import matplotlib.pyplot as plt


#####################################################################
#####################################################################

def main():

    """ EXTRACT DATA AT STATIONS LOCATIONS FROM EPICC RUNS AND COMPARE
        input: AEMET hifreq records, EPICC WRF runs
    """
    create_df = False
    obs_record,nstn_info = load_AEMET_hifreq()
    if create_df == True:
        ##EPICC files
        wrun = cfg.wrf_runs[0]

        fileref = nc.Dataset(f'{cfg.path_wrfout}/{wrun}/out/{cfg.file_ref}')
        filesin = sorted(glob(f'{cfg.path_postproc}/{wrun}/UIB_01H_RAIN_????-??.nc'))
        fin_all = xr.open_mfdataset(filesin,concat_dim="time", combine="nested")
        #wrf_pr10max = fin_all.resample(time=f"1H").sum('time').max(dim='time').load()
        wrf_pr10max = fin_all.max(dim='time').load()
        #Extract info and create df

        df = pd.DataFrame(columns=['wrf', 'obs'], index=nstn_info.INDICATIVO)
        for stn_id in nstn_info.INDICATIVO:
            x,y = ll_to_xy(fileref,nstn_info[nstn_info['INDICATIVO']==stn_id].NLAT,nstn_info[nstn_info['INDICATIVO']==stn_id].NLON)
            wrf_data = wrf_pr10max.sel(y=y,x=x)
            df['wrf'][stn_id]=wrf_data.RAIN.values
            df['obs'][stn_id]=obs_record.loc[obs_record.INDICATIVO==stn_id].PMAX60.max()/10.
        df.to_csv(f'{cfg.path_out}/Obs_comparison/PMAX60_WRFEPICC-AEMET.csv')

    else:
        df = pd.read_csv(f"{cfg.path_out}/Obs_comparison/PMAX60_WRFEPICC-AEMET.csv")

    df = df[df.obs>0]


    #####################################################################
    #####################################################################
    ## PLOTTING ##
    #####################################################################
    #####################################################################

    params = {'mathtext.default': 'regular' }
    plt.rcParams.update(params)

    fig = plt.figure(figsize=(15,7.5),constrained_layout=False)
    widths = [4,2]
    heights = [0.33]
    spec = fig.add_gridspec(ncols=2, nrows=1, width_ratios=widths,
                          height_ratios=heights)

    ax1 = fig.add_subplot(spec[0, 0])

    plt.scatter(df.index, df['obs'],s=100,c='k',edgecolor='face',alpha=0.7,label='OBS')
    plt.scatter(df.index, df['wrf'],s=100,c='#00A19D',edgecolor='face',alpha=0.7,label='WRF')
    ax1.axhline(y=df['obs'].mean(),color='k',linewidth=1.0)
    ax1.axhline(y=df['wrf'].mean(),color='#00A19D',linewidth=1.0)
    #plt.scatter(df.index, df['pgw'],s=100,c='#E05D5D',edgecolor='face',alpha=0.7,label='PGW')

    ax1.set_ylabel('Rainfall rate (mm $hr^{-1}$)')
    ax1.set_xlabel('Station no.')
    ax1.set_title('Comparison of hour max (2011-2020)',fontsize='xx-large')
    ax1.legend()

    ax2 = fig.add_subplot(spec[0, 1],sharey=ax1)

    bx = df.boxplot(ax=ax2,column=['obs', 'wrf'],rot=45,showfliers=False,return_type='dict',patch_artist=True,widths = 0.3,whis=[10,90])
    plt.setp(bx['boxes'][0],color='k',linewidth=0)
    plt.setp(bx['boxes'][1],color='#00A19D',linewidth=0)

    plt.setp(bx['boxes'][0],facecolor='k',alpha=0.7,edgecolor='k',linewidth=1.5)
    plt.setp(bx['boxes'][1],facecolor='#00A19D',alpha=0.7,edgecolor='#00A19D',linewidth=1.5)

    plt.setp(bx['caps'], color='none', linewidth=0.2)

    plt.setp(bx['medians'][0],color='k',linewidth=1.5)
    plt.setp(bx['medians'][1],color='#00A19D',linewidth=1.5)
    plt.setp(bx['whiskers'][0:2], color='gray', linewidth=1,ls='-')
    plt.setp(bx['whiskers'][2:4], color='#00A19D', linewidth=1,ls='-')

    ax2.axhline(y=0,color='k',linewidth=1.0)
    ax2.xaxis.set_ticklabels([])
    ax2.xaxis.set_ticks([])
    ax2.spines['top'].set_color('none')
    ax2.spines['right'].set_color('none')
    ax2.spines['bottom'].set_color('none')
    ax2.set_ylabel('mm',labelpad=2)


    #plt.show()
    plt.savefig(f'{cfg.path_out}/Obs_comparison/PRMAX60_WRFEPICC-AEMET_scatter.png',dpi=150)



#####################################################################
#####################################################################

def load_AEMET_hifreq():
    #AEMET files
    obs_path = '/vg6/dargueso-NO-BKUP/OBS_DATA/AEMET_MED_hifreq/'
    nstn_info = pd.read_csv(f'{obs_path}/AEMET_hifreq_station_info.csv')
    all_files = sorted(glob(f'{obs_path}/Arnau_*.txt'))     # advisable to use os.path.join as this makes concatenation OS independent
    df_from_each_file = (pd.read_csv(f, sep=';',header=0,encoding = "ISO-8859-1") for f in all_files)
    obs_record   = pd.concat(df_from_each_file, ignore_index=True)
    obs_record['dateInt']=obs_record['ANO'].astype(str) + obs_record['MES'].astype(str).str.zfill(2)+ obs_record['DIA'].astype(str).str.zfill(2)
    obs_record['Date'] = pd.to_datetime(obs_record['dateInt'], format='%Y%m%d')

    return obs_record,nstn_info
###############################################################################
##### __main__  scope
###############################################################################

if __name__ == "__main__":

    main()

###############################################################################
