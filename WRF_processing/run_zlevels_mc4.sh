#!/bin/bash
# Chain for mc4: temperature (TKZ, K) and pressure (PZ, Pa) at the fixed heights of
# EPICC_post_config.zlevels_m, August-November 2011-2020, for the present and PGW runs.
# TKZ and PZ run side by side (NJOBS workers each); the two runs one after the other.
# Usage: ./run_zlevels_mc4.sh [NJOBS]   (default 10)
# Launch detached:
#   (setsid nohup ./run_zlevels_mc4.sh > ~/postprocessed/EPICC/chain_zlevels.log 2>&1 < /dev/null &)
cd "$(dirname "$0")"
NJOBS=${1:-10}
P=$HOME/micromamba/envs/wrfproc/bin/python
LOGDIR=$HOME/postprocessed/EPICC
for WRUN in EPICC_2km_ERA5 EPICC_2km_ERA5_CMIP6anom; do
  echo "$(date '+%F %T') $WRUN: starting TKZ and PZ in parallel (Aug-Nov, $NJOBS workers each)"
  for VAR in TKZ PZ; do
    $P run_var_remote.py --wrun $WRUN --var $VAR --syear 2011 --eyear 2020 --months 8 9 10 11 --njobs $NJOBS > $LOGDIR/${VAR,,}_$WRUN.log 2>&1 &
  done
  wait
  for VAR in TKZ PZ; do
    echo "$(date '+%F %T') $WRUN: $VAR finished (done: $(grep -ac 'done in' $LOGDIR/${VAR,,}_$WRUN.log), errors: $(grep -ac ': ERROR' $LOGDIR/${VAR,,}_$WRUN.log))"
  done
done
echo "chain finished"
