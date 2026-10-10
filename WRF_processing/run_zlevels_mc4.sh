#!/bin/bash
# Chain for mc4: temperature (TKZ, K) and pressure (PZ, Pa) at the fixed heights of
# EPICC_post_config.zlevels_m for the present and PGW runs, TKZ and PZ side by side
# (NJOBS workers each), the two runs one after the other.
# Usage: ./run_zlevels_mc4.sh [NJOBS] [MONTHS...]   (default 10 workers, months 8 9 10 11)
# Launch detached, e.g. the remaining months:
#   (setsid nohup ./run_zlevels_mc4.sh 12 1 2 3 4 5 6 7 12 > ~/postprocessed/EPICC/chain_zlevels_rest.log 2>&1 < /dev/null &)
# The local copy loop (sync_zlevels_from_mc4.sh <logname>) waits for "chain finished" in that log.
cd "$(dirname "$0")"
NJOBS=${1:-10}; shift
MONTHS=${@:-8 9 10 11}
P=$HOME/micromamba/envs/wrfproc/bin/python
LOGDIR=$HOME/postprocessed/EPICC
for WRUN in EPICC_2km_ERA5 EPICC_2km_ERA5_CMIP6anom; do
  echo "$(date '+%F %T') $WRUN: starting TKZ and PZ in parallel (months $MONTHS, $NJOBS workers each)"
  for VAR in TKZ PZ; do
    $P run_var_remote.py --wrun $WRUN --var $VAR --syear 2011 --eyear 2020 --months $MONTHS --njobs $NJOBS > $LOGDIR/${VAR,,}_$WRUN.log 2>&1 &
  done
  wait
  for VAR in TKZ PZ; do
    echo "$(date '+%F %T') $WRUN: $VAR finished (done: $(grep -ac 'done in' $LOGDIR/${VAR,,}_$WRUN.log), errors: $(grep -ac ': ERROR' $LOGDIR/${VAR,,}_$WRUN.log))"
  done
done
echo "chain finished"
