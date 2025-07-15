import xarray as xr
import numpy as np

path_in = '/home/dargueso/postprocessed/EPICC/'
WET_VALUE_H = 0.1  # mm
WET_VALUE_D = 1.0 # mm
drain_bins = np.concatenate((np.arange(1, 10, 1), np.arange(10, 105, 5)))
hrain_bins = np.concatenate((np.arange(0, 10, 1), np.arange(10, 105, 5)))
buffer = 20
n_samples = 100

# Define quantiles for the analysis
qs = np.array([0.50, 0.75, 0.90, 0.95, 0.99, 0.999], dtype=np.float32)