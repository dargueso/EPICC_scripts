import xarray as xr
import numpy as np
import epicc_config as cfg
import time
import glob
import matplotlib as mpl
import matplotlib.pyplot as plt


def main():

    wrun_pres = 'EPICC_2km_ERA5'
    wrun_fut = 'EPICC_2km_ERA5_CMIP6anom'
    patt_in = 'UIB'
    fqs = ['10MIN','01H','DAY']
    percentiles = [90, 95, 99, 99.9, 99.99]  
    for fq in fqs:
        fpres = xr.open_dataset(f'{cfg.path_in}/{wrun_pres}/{patt_in}_{fq}_RAIN_TD2_percentiles.nc')
        ffut = xr.open_dataset(f'{cfg.path_in}/{wrun_fut}/{patt_in}_{fq}_RAIN_TD2_percentiles.nc')

        cpres = ["powderblue", "skyblue", "dodgerblue", "royalblue", "navy"]
        cfut = ["salmon", "tomato", "crimson", "firebrick", "darkred"]
        bincenter_pres = fpres.temp.values 
        bincenter_fut = ffut.temp.values
        bincenter = np.arange(200,306,1)

        dplot_pres_sig = fpres.RAIN.where(fpres.EVENT_COUNT*100/fpres.EVENT_COUNT.sum()>0.1).values
        dplot_fut_sig = ffut.RAIN.where(ffut.EVENT_COUNT*100/ffut.EVENT_COUNT.sum()>0.1).values
        dplot_pres_all = fpres.RAIN.values
        dplot_fut_all = ffut.RAIN.values

        
        mpl.style.use('seaborn-v0_8-paper')
        fig = plt.figure(figsize=(10,10)) 
        ax=fig.add_subplot(1,1,1)
        ax.set_ylabel('Precip (mm hr-1)')
        ax.set_xlabel('Dew Point Temp (\u00B0C)')
        for i in np.arange(230,320,2):
            ax.plot(bincenter,1.07**(bincenter-i)*5, c='lightgray', linewidth=0.7,linestyle='--')
            ax.plot(bincenter,1.14**(bincenter-i)*5, c='gray', linewidth=0.5,linestyle='--')

        for nqt, qt in enumerate(percentiles):

            ax.plot(bincenter_pres,dplot_pres_all[:,nqt],color=cfut[nqt], label=f'{qt}%', linestyle=':', linewidth=0.5)
            ax.plot(bincenter_fut,dplot_fut_all[:,nqt],color=cfut[nqt], label=f'{qt}%', linestyle=':', linewidth=0.5)
            ax.plot(bincenter_pres,dplot_pres_sig[:,nqt],color=cpres[nqt], label=f'{qt}%', linestyle='-', linewidth=2)
            ax.plot(bincenter_fut,dplot_fut_sig[:,nqt],color=cfut[nqt], label=f'{qt}%', linestyle='-', linewidth=2)

            ax.annotate(qt,(bincenter_fut[np.nanargmax(dplot_fut_sig[:,nqt])],np.nanmax(dplot_fut_sig[:,nqt])),xytext=(0,10 ),ha='right',
            va='center',color=cfut[nqt],textcoords='offset points',family='monospace')

        ax.set_xlim(275,bincenter.max())
        ax.set_ylim(1,1000)
        ax.set_yscale("log")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.savefig(f'./EPICC_{patt_in}_{fq}_RAIN_TD2_percentiles_pres_fut.png',dpi=300)





###############################################################################
# __main__  scope
###############################################################################


if __name__ == "__main__":
    raise SystemExit(main())

###############################################################################
