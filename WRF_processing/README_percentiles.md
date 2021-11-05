# Calculation of percentiles (tile approach)

The calculation of percentiles is very demanding in terms of memory if the postprocessed files are used, because it needs to load the entire period. 
To optimize the calculation we need to take a few prior steps:
1. Postprocess 
2. Split files into lat-lon tiles
3. Calculate percentiles for each of the tiles
4. Load and merge

All scripts are located in the WRF_processing  directory within the EPICC_scripts repository. 

Step 1 - Postprocessed is done with  EPICC_postprocess_parallel.py, which uses EPICC_post_config.py to decide on postprocessing options.
Step 2 - Split files into lat-lon tiles is done with split_files_tiles_latlon.py, which needs epicc_config.py (in the repository parent folder).  We can specify the tile_size in the script in number of grid points.
Step 3 - The percentiles are calculated using create_percentiles_split_files_tiles_latlon.py. It requies a few options to be set. The frequency, defines the original files to be read. The mode (wetonly or all values) and the threshold to define a wet value. Some are defined in the script itself, some other are defined in the epicc_config.py file. The tile_size needs to be consistent with step 2. This generates a file with all requested percentiles for each tile.

Step 4 - To create a file with percentiles over the entire domain, we need to load and merge files from step 3 using load_and_merge_files_tiles_latlon.py. This script also need some options to be set, mostly to define which files will be loaded and merge (tile_size, mode, wrun).