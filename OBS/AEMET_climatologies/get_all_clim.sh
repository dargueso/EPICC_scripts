#!/bin/bash
source /Users/daniel/opt/anaconda3/etc/profile.d/conda.sh
conda activate todayextreme
python -W ignore /Users/daniel/Scripts_local/EPICC_scripts/OBS/AEMET_climatologies/get_and_fill_climatologies_allavailable.py
conda deactivate
#rsync /home/dargueso/todayextreme/locations.json /disk17/people/dargueso/
#rsync -a --delete /home/dargueso/todayextreme/Data/plots/* /disk17/people/dargueso/output/
