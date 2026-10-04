#!/bin/bash
# Second stage for mc4: once the present-run TKZ/PZ jobs have exited, run TKZ and PZ for the
# PGW run (EPICC_2km_ERA5_CMIP6anom), August-November 2011-2020, side by side with NJOBS
# workers each (more than the 4 used while WAZ was still running). Appends "chain finished"
# to chain_zlevels.log, which the local copy loop waits for.
# Usage: ./run_zlevels_cmip_mc4.sh [NJOBS]   (default 10)
cd "$(dirname "$0")"
NJOBS=${1:-10}
P=$HOME/micromamba/envs/wrfproc/bin/python
LOGDIR=$HOME/postprocessed/EPICC
WRUN=EPICC_2km_ERA5_CMIP6anom
while pgrep -u $USER -f "wrun EPICC_2km_ERA5 --var (TKZ|PZ)" > /dev/null; do sleep 120; done
echo "$(date '+%F %T') $WRUN: starting TKZ and PZ in parallel (Aug-Nov, $NJOBS workers each)" | tee -a $LOGDIR/chain_zlevels.log
for VAR in TKZ PZ; do
  $P run_var_remote.py --wrun $WRUN --var $VAR --syear 2011 --eyear 2020 --months 8 9 10 11 --njobs $NJOBS > $LOGDIR/${VAR,,}_$WRUN.log 2>&1 &
done
wait
for VAR in TKZ PZ; do
  echo "$(date '+%F %T') $WRUN: $VAR finished (done: $(grep -ac 'done in' $LOGDIR/${VAR,,}_$WRUN.log), errors: $(grep -ac ': ERROR' $LOGDIR/${VAR,,}_$WRUN.log))" | tee -a $LOGDIR/chain_zlevels.log
done
echo "chain finished" | tee -a $LOGDIR/chain_zlevels.log
