import xarray as xr
import numpy as np
import pandas as pd
import glob
import epicc_config as cfg
import time
from memory_profiler import profile
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
import joblib
import os
import dask.array as da

def read_pkl(filename):
    import pickle as pkl
    handle=open('%s' % filename,'rb')
    var=pkl.load(handle)
    return var
def process_year_month(year, month, cfg, wrun, patt_in, fq, mask_stacked):
    print(f'Processing year {year} and month {month}')

    if fq == '01H':

        filepr = f'{cfg.path_in}/{wrun}/{patt_in}_{fq}_RAIN_{year}-{month:02d}.nc'
        filetd2 = f'{cfg.path_in}/{wrun}/{patt_in}_{fq}_TD2_{year}-{month:02d}.nc'

        if not os.path.exists(filepr) or not os.path.exists(filetd2):
            print(f"Skipping {year}-{month:02d} (files not found)")
            return

        ds_rain = xr.open_dataset(filepr)
        ds_td2 = xr.open_dataset(filetd2)

    elif fq == '10MIN':

        filepr = f'{cfg.path_in}/{wrun}/bymonth/compressed/{patt_in}_{fq}_RAIN_{year}-{month:02d}.nc'
        filetd2 = f'{cfg.path_in}/{wrun}/{patt_in}_01H_TD2_{year}-{month:02d}.nc'

        if not os.path.exists(filepr) or not os.path.exists(filetd2):
            print(f"Skipping {year}-{month:02d} (files not found)")
            return

        ds_rain = xr.open_dataset(filepr).resample(time="1h").max()
        ds_td2 = xr.open_dataset(filetd2)

    elif fq == 'DAY':

        filepr = f'{cfg.path_in}/{wrun}/{patt_in}_01H_RAIN_{year}-{month:02d}.nc'
        filetd2 = f'{cfg.path_in}/{wrun}/{patt_in}_01H_TD2_{year}-{month:02d}.nc'

        if not os.path.exists(filepr) or not os.path.exists(filetd2):
            print(f"Skipping {year}-{month:02d} (files not found)")
            return

        ds_rain = xr.open_dataset(filepr).resample(time="1D").sum()
        ds_td2 = xr.open_dataset(filetd2).resample(time="1D").max()

    ds_rain = ds_rain.drop_vars("time", errors="ignore")
    ds_td2 = ds_td2.drop_vars("time_bnds", errors="ignore")
    ds_rain = ds_rain.drop_vars("time_bnds", errors="ignore")

    ds = xr.merge([ds_rain, ds_td2])

    ds_masked = ds.stack(loc=("y", "x")).where(mask_stacked, drop=True).reset_index("loc")
    masked_file = f"{cfg.path_in}/{wrun}/{patt_in}_{fq}_RAIN_TD2_{year}-{month:02d}_masked.nc"
    ds_masked.to_netcdf(masked_file)

    ds_masked_stacked = ds_masked.stack(event=("time", "loc")).reset_index("event")
    ds_masked_wet = ds_masked_stacked.where(ds_masked_stacked.RAIN > cfg.wet_value, drop=True)
    
    masked_wet_file = f"{cfg.path_in}/{wrun}/{patt_in}_{fq}_RAIN_TD2_{year}-{month:02d}_masked_wet.nc"
    ds_masked_wet.to_netcdf(masked_wet_file)

    # Cleanup
    del ds, ds_masked, ds_rain, ds_td2, ds_masked_stacked, ds_masked_wet

#First we need to load postprocessed files for RAIN and TD2
#We will merge them into a single file and mask the data to reduce size
#We will then extract only rain days to further reduce size

@profile
def main():
    fq = 'DAY'
    wrun = 'EPICC_2km_ERA5'
    patt_in = 'UIB'
    bincenter_one=list(range(264,311,1))
    colors=['indigo','lightseagreen', 'yellowgreen','orange','crimson']#indigo removed

    mask = read_pkl('./mediterranean_coastal_mask_250km.pkl')
    mask_xr = xr.DataArray(mask, dims=("y", "x"), name="mask")
    mask_stacked = mask_xr.stack(loc=("y", "x"))

    create_masked_files = False
    calc_percentiles = True
    plot_scaling = True

    if create_masked_files:

        # Number of parallel jobs (adjust based on available cores)
        N_JOBS = 12

        # Run in parallel
        joblib.Parallel(n_jobs=N_JOBS)(
            joblib.delayed(process_year_month)(year, month, cfg, wrun, patt_in, fq, mask_stacked)
            for year in range(cfg.syear, cfg.eyear + 1)
            for month in range(1, 13)
        )


    if calc_percentiles:
        
        files = sorted(glob.glob(f'{cfg.path_in}/{wrun}/{patt_in}_{fq}_RAIN_TD2_*_masked_wet.nc'))

        # Open multiple files using dask for parallelism
        ds = xr.open_mfdataset(files, combine='nested', concat_dim='event', chunks={"event": 100000})

        # Extract variables
        td2 = ds["TD2"]  # Temperature at 2m
        rain = ds["RAIN"]  # Rainfall

        # Define temperature bins (assuming TD2 is in Kelvin)
        temp_bins = np.arange(264, 311, 1)  # 1K bins from 264 to 310
        bin_labels = temp_bins[:-1]  # Label bins by their lower bound

        # Assign each event to a bin
        bin_indices = np.digitize(td2, temp_bins) - 1  # Convert to zero-based index

        # Percentiles to compute
        percentiles = [90, 95, 99, 99.9, 99.99]  

        results = {}
        nevents = {}

        # Compute percentiles for each bin
        for i, temp in enumerate(bin_labels):
            # Create the mask for the current bin
            mask = xr.DataArray(bin_indices == i, coords=rain.coords, dims=rain.dims)
            rain_in_bin = rain.where(mask, drop=True)
            event_count = rain_in_bin.count().compute()  # Compute event count

            if event_count > 0:  # Compute count to check for data presence
                results[temp] = da.percentile(rain_in_bin.data, percentiles).compute()  # Compute percentiles
                nevents[temp] = event_count

        # Convert results dictionary to DataFrame
        df_percentiles = pd.DataFrame.from_dict(results, orient="index", columns=[f"P{p}" for p in percentiles])
        df_percentiles.index.name = "temp"
        df_nevents = pd.DataFrame.from_dict(nevents, orient="index", columns=["count"])
        df_nevents.index.name = "temp"


        ds_percentiles = xr.DataArray(
            df_percentiles.values, 
            dims=("temp", "percentiles"), 
            coords={"temp": df_percentiles.index, "percentiles": percentiles},
            name="RAIN"
        )

        ds_nevents = xr.DataArray(
        df_nevents.values.flatten(), 
        dims=("temp",), 
        coords={"temp": df_nevents.index},
        name="EVENT_COUNT"
        )

        ds_out = xr.Dataset({"RAIN": ds_percentiles, "EVENT_COUNT": ds_nevents})
        ds_out.to_netcdf(f'{cfg.path_in}/{wrun}/{patt_in}_{fq}_RAIN_TD2_percentiles.nc')

    if plot_scaling:
        print('Plotting scaling')

        fin = xr.open_dataset(f'{cfg.path_in}/{wrun}/{patt_in}_{fq}_RAIN_TD2_percentiles.nc')
        bincenter=fin.temp.values
        dataplotpr = fin.RAIN.values
        mpl.style.use('seaborn-v0_8-paper')   
        fig = plt.figure(figsize=(10,10)) 
        ax=fig.add_subplot(1,1,1)
        ax.set_ylabel('Precip (mm hr-1)')
        ax.set_xlabel('Dew Point Temp (\u00B0C)')

        for i in np.arange(230,320,2):
            ax.plot(bincenter,1.07**(bincenter-i)*5, c='lightgray', linewidth=0.5,linestyle='--')
            ax.plot(bincenter,1.14**(bincenter-i)*5, c='gray', linewidth=0.5,linestyle='--')

        for nqt, qt in enumerate(cfg.qtiles):
            ax.plot(bincenter,dataplotpr[:,nqt],color=colors[nqt], label=f'{qt}%')
            ax.annotate(qt,(bincenter[np.nanargmax(dataplotpr[:,nqt])],np.nanmax(dataplotpr[:,nqt])),xytext=(0,10 ),ha='right',
                va='center',color=colors[nqt],textcoords='offset points',family='monospace')

        ax.set_xlim(275,bincenter.max())
        ax.set_ylim(5,1000)
        ax.set_yscale("log")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.savefig(f'./{wrun}_{patt_in}_{fq}_RAIN_TD2_percentiles.png',dpi=300)
    #####################################################################
    #####################################################################
###############################################################################
# __main__  scope
###############################################################################


if __name__ == "__main__":
    raise SystemExit(main())

###############################################################################
