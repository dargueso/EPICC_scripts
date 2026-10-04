#!/bin/bash
# Chain for mc4: SST from hourly wrfout, then WAZ (15 levels MSL) from wrf3hrly.
# Usage:  ./run_sst_then_waz_mc4.sh [WRUN]      (default EPICC_2km_ERA5)
# Launch detached:
#   (setsid nohup ./run_sst_then_waz_mc4.sh EPICC_2km_ERA5_CMIP6anom > ~/postprocessed/EPICC/chain_sst_waz_EPICC_2km_ERA5_CMIP6anom.log 2>&1 < /dev/null &)
cd "$(dirname "$0")"
WRUN=${1:-EPICC_2km_ERA5}
P=$HOME/micromamba/envs/wrfproc/bin/python
LOGDIR=$HOME/postprocessed/EPICC
echo "$(date '+%F %T') $WRUN: starting SST"
$P run_var_remote.py --wrun $WRUN --var SST --patt wrfout --syear 2011 --eyear 2020 --njobs 16 > $LOGDIR/sst_$WRUN.log 2>&1
echo "$(date '+%F %T') SST finished (errors: $(grep -ac ': ERROR' $LOGDIR/sst_$WRUN.log)); starting WAZ"
$P run_var_remote.py --wrun $WRUN --var WAZ --syear 2011 --eyear 2020 --njobs 16 > $LOGDIR/waz_$WRUN.log 2>&1
echo "$(date '+%F %T') WAZ finished (errors: $(grep -ac ': ERROR' $LOGDIR/waz_$WRUN.log))"
echo "chain finished"
