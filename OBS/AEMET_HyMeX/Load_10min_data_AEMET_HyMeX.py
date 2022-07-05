#!/usr/bin/env python
import pandas as pd
import datetime as dt
import numpy as np



for year in range(2011,2021):
    dates = pd.date_range(dt.datetime(year,1,1,0,0),dt.datetime(year,12,31,23,50),freq='10T')
    df_prec = pd.DataFrame(columns=[''],index=dates)
    for month in range(1,13):


        filein = f"./{year}/AWS_AEMET-{year}{month:02d}.txt"
        print(f"{filein}")
        df = pd.read_csv(filein, header=0, sep=' ',na_values=-999.0,parse_dates=['DateTime'])
        df = df.rename(columns={"#Code":"stn_id"})
        df = df.set_index('DateTime')
        stations = df.stn_id.unique()
        for stat in stations:
            if stat not in df_prec:
                df_prec[stat]=np.nan
            df_stat = df[df.stn_id==stat].Prec
            df_prec = df_prec.combine_first(df_stat.to_frame().rename(columns={'Prec':stat}))
    df_prec.to_pickle(f"prec_10min_all_stations_{year}.pkl")

