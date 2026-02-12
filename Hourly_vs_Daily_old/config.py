import xarray as xr
import numpy as np

path_in = '/home/dargueso/postprocessed/EPICC/'
path_out = '/home/dargueso/postprocessed/EPICC/'
WET_VALUE_H = 0.1  # mm
WET_VALUE_D = 1.0 # mm
drain_bins = np.arange(0, 105, 5)
hrain_bins = np.arange(0, 101, 1)
buffer =10 
tile_size = 50
n_samples = 1000
syear = 2011
eyear = 2020

# Define quantiles for the analysis
qs = np.array([0.50, 0.75, 0.90, 0.95, 0.99, 0.999], dtype=np.float32)
