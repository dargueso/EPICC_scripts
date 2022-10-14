#!/usr/bin/env python
"""
#####################################################################
# Author: Daniel Argueso <daniel>
# Date:   2020-02-09T19:12:47+01:00
# Email:  d.argueso@uib.es
# Last modified by:   daniel
# Last modified time: 2020-02-09T19:12:47+01:00
#
# @Project@
# Version: x.0 (Beta)
# Description:
#
# Dependencies:
#
# Files:
#
#####################################################################
"""



from aemet import Aemet, Estacion
import json
import os
import pandas as pd
import numpy as np
import xarray as xr
import datetime as dt
import os
import unidecode

from glob import glob

###########################################################
############# USER MODIF ##################################

syear = 1920
eyear = 2022
year_ref=2021
nyears = eyear-syear+1
path_in = "/Users/daniel/Scripts_local/EPICC_scripts/OBS/AEMET_climatologies/Data_records/"
rewrite = False
getfiles = True
fillfiles = False
concatfiles = False
############# END OF USER MODIF ###########################
###########################################################

aemet = Aemet()

with open('/Users/daniel/Scripts_local/EPICC_scripts/OBS/AEMET_climatologies/info_estaciones_climavail.json') as loc_file:
    locations = json.load(loc_file)

df_locs = pd.DataFrame.from_dict(locations)

if ('syear_download' not in df_locs):
    df_locs['syear_download'] = None
if ('start_year' not in df_locs):
    df_locs['start_year'] = None
if ('end_year' not in df_locs):
    df_locs['end_year'] = None



stations = df_locs.indicativo

###########################################################
###########################################################

##GETTING CLIMATOLOGIES FROM AEMET

if getfiles:
    for stn_id in stations:
        print(stn_id)
        nsinfo = df_locs[df_locs.indicativo==stn_id]
        nsname = nsinfo['nombre'].item().title()
        stn_url = nsname.replace(","," ").replace("/","_").replace(" ","_").replace("__","_").lower()        
	
        stn_url = unidecode.unidecode(stn_url)


        print (f"Getting station {nsname}")
        if rewrite: 
            print(f'Record files will be rewritten if they exist')

        #First time - create folder for station
        if not os.path.exists(f'{path_in}/{stn_url}'):
            os.makedirs(f'{path_in}/{stn_url}')

        #Check downloading progress in station
 
        if (df_locs[df_locs.indicativo==stn_id].syear_download.notnull().any()):
            syear_download = int(df_locs[df_locs.indicativo==stn_id]['syear_download'])
        else:
            syear_download = int(syear)
            df_locs.loc[df_locs.indicativo==stn_id,'syear_download']=int(syear)

        
        for yr in range(syear_download , eyear+1):
            filename = f'{path_in}/{stn_url}/Data_ClimDay_{stn_url}_{yr}.csv'
        
            if ((os.path.exists(filename)) and (not rewrite)):
                df_locs.loc[df_locs.indicativo==stn_id,'syear_download']=int(yr)+1
                continue
        
            #Downloading year and writing record
            sdate = f'{yr}-01-01T00:00:00UTC'
            edate = f'{yr}-12-31T00:00:00UTC'

            try:
                vcm = aemet.get_valores_climatologicos_diarios(sdate,edate, stn_id)
            except:
                vcm = aemet.get_valores_climatologicos_diarios(sdate,edate, stn_id)
            
            if isinstance(vcm, dict):
                if 'descripcion' in vcm.keys():
                    if vcm['descripcion']=='No hay datos que satisfagan esos criterios':
                        print(f'No data for year {yr}')
                        os.popen(f'cp empty.csv {filename}')
                        df_locs.loc[df_locs.indicativo==stn_id,'start_year']=int(yr)+1
                        print(f'Created empty record {stn_id} year {yr}')

            else:


                for rc in range(len(vcm)):
                    if 'prec' in vcm[rc].keys():
                        if vcm[rc]['prec']=='Ip':
                            vcm[rc]['prec']='0,0'
                        if vcm[rc]['prec']=='Acum':
                            vcm[rc]['prec']=None
                # vcm = {k['prec']: '0,0' if v['prec']=='Ip' else v['prec'] for k, v in vcm}
                # vcm = {k['prec']: None  if v['prec']=='Acum' else v['prec'] for k, v in vcm}
    
                data_pd = pd.DataFrame(vcm)
                data_pd.to_csv(filename)
                print(f'Succesfully downloaded {stn_id} year {yr}')

            df_locs.loc[df_locs.indicativo==stn_id,'syear_download']=int(yr)+1
            with open('/Users/daniel/Scripts_local/EPICC_scripts/OBS/AEMET_climatologies/info_estaciones_climavail.json', 'w') as outjsn:
                json.dump(df_locs.to_dict(orient='records'), outjsn, indent=1,ensure_ascii=False)

  

###########################################################
###########################################################

#FILLING CLIMATOLOGIES AND FIXING
if fillfiles:
    for stn_id in stations:
        nsinfo = df_locs[df_locs.indicativo==stn_id]
        nsname = nsinfo['nombre'].item().title()

        stn_url = nsname.replace(","," ").replace(" ","_").lower()

        print (f"Filling station {nsname}")
        today=dt.date.today()
        fyear = True
        for year in range(syear,eyear+1):
            fileref = f'{path_in}/{stn_url}/Data_ClimDay_{stn_url}_{year_ref}.csv'
            filename = f'{path_in}/{stn_url}/Data_ClimDay_{stn_url}_{year}.csv'
            filenew = f'{path_in}/{stn_url}/Data_ClimDay_{stn_url}_{year}_new.csv'

            if (fyear == True) & (~os.path.exists(filename)):
                continue
            else:
                fyear = False

            fref = pd.read_csv(fileref, index_col=None, sep=',',decimal=",")


            if year == today.year:
                fthis = pd.read_csv(filename, index_col=None, sep=',',decimal=",")
                date_index = pd.date_range(f'{year}-01-01',f'{fthis["fecha"].iloc[-1]}')
                fnew = pd.DataFrame(np.ones((len(date_index),len(fref.columns)))*np.nan,columns=fref.columns)
                fnew['fecha']=[date_index[i].strftime("%Y-%m-%d") for i in range(len(date_index))]
            else:
                date_index = pd.date_range(f'{year}-01-01',f'{year}-12-31')
                fnew = pd.DataFrame(np.ones((len(date_index),len(fref.columns)))*np.nan,columns=fref.columns)
                fnew['fecha']=[date_index[i].strftime("%Y-%m-%d") for i in range(len(date_index))]


            print (f'Filling year {year}')
            if os.path.exists(filename):
                fthis = pd.read_csv(filename, index_col=None, sep=',',decimal=",")
                if len(fthis)!=len(date_index):print (f'Filling incomplete year {year}')



                for n,day in enumerate(fthis.fecha):
                    for f,field in enumerate(fnew.columns):
                        if field in fthis.columns:
                            if field in ['indicativo','nombre','provincia','altitud']:
                                continue
                            else:
                                fnew.loc[fnew.fecha==day,field]=fthis.loc[fthis.fecha==day,field].values

            else:
                print (f'Filling non-existent year {year}')

            fnew['fecha']
            fnew['indicativo']=stn_id
            fnew['nombre']=fref.loc[0,'nombre']
            fnew['provincia']=fref.loc[0,'provincia']
            fnew['altitud']=fref.loc[0,'altitud']
            del fnew['Unnamed: 0']
            fnew.to_csv(filenew)

###########################################################
###########################################################

### CONCATENATE CLIM INTO A SINGLE CSV
if concatfiles:
    for stn_id in stations:

        nsinfo = df_locs[df_locs.indicativo==stn_id]
        nsname = nsinfo['nombre'].item().title()
        stn_url = nsname.replace(","," ").replace(" ","_").lower()

        all_files = sorted(glob(f'{path_in}/{stn_url}/Data_ClimDay_{stn_url}_*_new.csv'))
        li = []

        for filename in all_files:
            df = pd.read_csv(filename, index_col=None, sep=',')
            li.append(df)
        clim_data = pd.concat(li, axis=0, ignore_index=True,sort=False)

        clim_data = clim_data.set_index('fecha')
        clim_data = clim_data.rename_axis('datetime')
        clim_data.index = pd.to_datetime(clim_data.index.get_level_values(0).astype(str),format='%Y-%m-%d')
        clim_data.to_csv(f'{path_in}/{stn_url}/Data_ClimDay_{stn_url}_allrecord.csv')
