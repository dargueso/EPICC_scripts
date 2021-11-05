import pandas as pd
from glob import glob

#####################################################################
#####################################################################

#AEMET files
all_files = sorted(glob('/vg6/dargueso-NO-BKUP/OBS_DATA/AEMET_MED_hifreq/Arnau_*.txt'))     # advisable to use os.path.join as this makes concatenation OS independent
df_from_each_file = (pd.read_csv(f, sep=';',header=0,encoding = "ISO-8859-1") for f in all_files)
prec_data   = pd.concat(df_from_each_file, ignore_index=True)
prec_data['dateInt']=prec_data['ANO'].astype(str) + prec_data['MES'].astype(str).str.zfill(2)+ prec_data['DIA'].astype(str).str.zfill(2)
prec_data['Date'] = pd.to_datetime(prec_data['dateInt'], format='%Y%m%d')
stn_info = prec_data.take(prec_data['INDICATIVO'].drop_duplicates().index)


#####################################################################
#####################################################################

stn_info["NLAT"] = 0
stn_info["NLON"] = 0
stn_info["sdate"] = 0
stn_info["edate"] = 0
nstn_info = stn_info[['INDICATIVO','NOMBRE','ALTITUD','NLON','NLAT','NOM_PROV','sdate','edate']]


for i, row in stn_info.iterrows():

    stn = row.INDICATIVO
    
    lon_str = f'{row.LONGITUD:07}'
    lat_str = f'{row.LATITUD:06}'
    
    if int(lon_str[-1])==1:
        new_lon = float(lon_str[:2])+float(lon_str[2:4])/60.+float(lon_str[4:6])/3600.
    elif int(lon_str[-1])==2:
        new_lon = -(float(lon_str[:2])+float(lon_str[2:4])/60.+float(lon_str[4:6])/3600.)

    new_lat = float(lat_str[:2])+float(lat_str[2:4])/60.+float(lat_str[4:6])/3600.
    
    nstn_info.at[nstn_info.INDICATIVO==stn,'NLON'] = round(new_lon,3)
    nstn_info.at[nstn_info.INDICATIVO==stn,'NLAT'] = round(new_lat,3)
    nstn_info.at[nstn_info.INDICATIVO==stn,'sdate']= prec_data[prec_data.INDICATIVO==stn].Date.min()
    nstn_info.at[nstn_info.INDICATIVO==stn,'edate'] = prec_data[prec_data.INDICATIVO==stn].Date.max()

nstn_info = nstn_info.reset_index(drop=True)
nstn_info.to_csv('/vg6/dargueso-NO-BKUP/OBS_DATA/AEMET_MED_hifreq//AEMET_hifreq_station_info.csv',index=False)

