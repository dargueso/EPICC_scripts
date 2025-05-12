#!/usr/bin/env python
'''
@File    :  Untitled-1
@Time    :  2025/03/29 09:51:18
@Author  :  Daniel Argüeso
@Version :  1.0
@Contact :  d.argueso@uib.es
@License :  (C)Copyright 2022, Daniel Argüeso
@Project :  None
@Desc    :  The data was previously processed with CDO:
First we calculate the daily maximum of the 10min data:
for file in $(ls UIB_10MIN_RAIN_????-??.nc); do cdo daymax ${file} ${file/.nc/_daymax.nc}; done
Then we concatenate files to get one file per year:
for year in $(seq 2011 2020); do cdo cat UIB_10MIN_RAIN_${year}-??_daymax.nc UIB_10MIN_RAIN_${year}_daymax.nc; done
Then we calculate the annual maximum of the daily maximum:
for year in $(seq 2011 2020); do cdo timmax UIB_10MIN_RAIN_${year}_daymax.nc UIB_10MIN_RAIN_${year}_annual_max.nc; done
'''

import xarray as xr
import scipy.cluster.hierarchy as sch
import numpy as np
import matplotlib.pyplot as plt
import fastcluster 
import pandas as pd
import pickle
import epicc_config as cfg
from glob import glob
from scipy.cluster.hierarchy import inconsistent

wruns = ['EPICC_2km_ERA5']#,'EPICC_2km_ERA5_CMIP6anom']
#wrunp = wruns[0]
#wrunf = wruns[1]

calculate_linkage_matrix = False

linkage_matrix_file = "linkage_matrix.pkl"

#####################################################################
####################################################################


def main():
    
    
    # Using Ward's method, we will cluster the data 
    for wrun in wruns:
        filesin = sorted(glob(f'{cfg.path_in}/{wrun}/{cfg.patt_in}_10MIN_RAIN_????_annual_max.nc'))
        finp = xr.open_mfdataset(filesin, concat_dim="time", combine="nested").load()
        rainfall = finp.RAIN.values - finp.RAIN.min('time').values 
        
        #Load pkl with mask
        geo_file = xr.open_dataset(cfg.geofile_ref)
        mask=geo_file.LANDMASK.values.squeeze()
        #mask = pd.read_pickle(f'mediterranean_coastal_mask_250km.pkl')
        
        #Mask borders out
        mask[:80, :] = False   # Top border
        mask[-80:, :] = False  # Bottom border
        mask[:, :80] = False   # Left border
        mask[:, -80:] = False  # Right border

        
        rainfall = np.where(mask==1, rainfall, np.nan)
        time_steps, lat_size, lon_size = rainfall.shape
        rainfall_reshaped = rainfall.reshape(time_steps, -1).T
        
        
        rainfall_reshaped = rainfall_reshaped[~np.isnan(rainfall_reshaped).any(axis=1)]
        if calculate_linkage_matrix:
            linkage_matrix = fastcluster.linkage_vector(rainfall_reshaped, method="ward")
            
            # Assume `linkage_matrix` is already computed
            # Save using pickle
            with open(linkage_matrix_file, "wb") as f:
                pickle.dump(linkage_matrix, f)

            print(f"Linkage matrix saved to {linkage_matrix_file}")
            
        else:
            # Load the linkage matrix
            with open(linkage_matrix_file, "rb") as f:
                linkage_matrix = pickle.load(f)

            print("Linkage matrix loaded successfully!")
        
        #linkage_matrix = sch.ward(rainfall_reshaped)
        # Plot dendrogram
        plt.figure(figsize=(10, 5))
        sch.dendrogram(linkage_matrix, truncate_mode="level", p=10)
        plt.title("Ward's Hierarchical Clustering of Rainfall Data")
        plt.xlabel("Spatial Points")
        plt.ylabel("Distance")
        plt.savefig(f"ward_dendrogram_{wrun}.png")
        
        plt.close()
        
        
        #Use the Inconsistency Criterion
        depth = 20  # Number of levels to consider
        incons = inconsistent(linkage_matrix, depth)
        print(incons[:50])  # Inspect inconsistency values
        # Perform clustering
        # Define number of clusters (e.g., 4)
        
        #Use the Elbow Method on Dendrogram Distances
        last_merges = linkage_matrix[-100:, 2]  # Last 15 merges
        diffs = np.diff(last_merges)
        optimal_clusters = np.argmax(diffs) + 1  # +1 since diffs reduces dimension by 1

        print(f"Optimal Number of Clusters: {optimal_clusters}")

        #Use maxclust Based on Threshold
        distance_threshold = 500  # Adjust based on dendrogram
        cluster_labels = sch.fcluster(linkage_matrix, distance_threshold, criterion="distance")
        print(f"Number of clusters: {len(np.unique(cluster_labels))}")
        
        num_clusters = len(np.unique(cluster_labels))
        cluster_labels = sch.fcluster(linkage_matrix, num_clusters, criterion="maxclust")

        # Reshape back to (lat, lon)
        clusters_map =  np.zeros((lat_size, lon_size))
        clusters_map[mask == 1] = cluster_labels
        #clusters_map = cluster_labels.reshape(lat_size, lon_size)

        # Plot cluster map
        plt.figure(figsize=(8, 6))
        plt.imshow(clusters_map[::-1,:], cmap="tab10", origin="upper")
        plt.colorbar(label="Cluster ID")
        plt.title(f"Spatial Clustering of Rainfall Data ({num_clusters} Clusters)")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.savefig(f"ward_clusters_{wrun}.png")
        
   
        # Create an xarray DataArray from the clusters_map()
        cluster_da = xr.DataArray(clusters_map, dims=["y", "x"])
        cluster_da.name = "ClusterID"
        cluster_da.attrs["description"] = "Cluster ID from Ward's hierarchical clustering"
        cluster_da.attrs["units"] = "ID"
        cluster_da.attrs["num_clusters"] = num_clusters
        cluster_da.attrs["method"] = "Ward's hierarchical clustering"
        cluster_da.attrs["wrun"] = wrun
        cluster_da.attrs["date"] = str(np.datetime64('now', 's'))
        
        # Save the cluster map to a NetCDF file
        cluster_da.to_netcdf(f"ward_clusters_{wrun}_{num_clusters}.nc")
 

###############################################################################
##### __main__  scope
###############################################################################

if __name__ == "__main__":

    main()


###############################################################################
