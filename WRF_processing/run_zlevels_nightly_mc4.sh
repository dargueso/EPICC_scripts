#!/bin/bash
# Nightly window scheduler for mc4, as agreed with the other users (2026-10-06):
#   * every night at START (23:00) check how busy mc4 is over 30 s; if CPU busy > BUSY_MAX (80 %)
#     wait RETRY_MIN (15) minutes and try again, RETRIES (3) more times (23:15, 23:30, 23:45);
#     if still busy, skip the night;
#   * otherwise run the variables in VARS side by side with NJOBS (12) workers each at nice 10;
#   * no new day is submitted after STOP (04:00); days already running finish;
#   * repeat until every day of MONTHS exists for every variable, then append "chain finished"
#     to chain_zlevels_rest.log (watched by the local copy loop / finaliser).
# Usage (all optional, as environment variables):
#   NJOBS=24 STOP=05:00 WRUN=EPICC_2km_ERA5_CMIP6anom VARS="TKZ PZ" MONTHS="1 2 3 4 5 6 7 12" EXPECTED=2433 \
#   START=23:00 BUSY_MAX=80 RETRIES=3 RETRY_MIN=15 MEM_PER_WORKER_GB=22 MEM_FRACTION=85 ./run_zlevels_nightly_mc4.sh
# Launch detached:
#   (setsid nohup ./run_zlevels_nightly_mc4.sh > ~/postprocessed/EPICC/nightly_scheduler.out 2>&1 < /dev/null &)
cd "$(dirname "$0")"
NJOBS=${NJOBS:-12}; WRUN=${WRUN:-EPICC_2km_ERA5_CMIP6anom}; VARS=${VARS:-"TKZ PZ"}
MONTHS=${MONTHS:-"1 2 3 4 5 6 7 12"}; EXPECTED=${EXPECTED:-2433}   # days in MONTHS over 2011-2020
START=${START:-23:00}; STOP=${STOP:-04:00}; BUSY_MAX=${BUSY_MAX:-80}; RETRIES=${RETRIES:-3}; RETRY_MIN=${RETRY_MIN:-15}
MEM_PER_WORKER_GB=${MEM_PER_WORKER_GB:-22}; MEM_FRACTION=${MEM_FRACTION:-85}   # memory guard: cap workers to MEM_FRACTION % of MemAvailable
P=$HOME/micromamba/envs/wrfproc/bin/python; L=$HOME/postprocessed/EPICC
CHAIN=$L/chain_zlevels_rest.log; LOG=$L/nightly_scheduler.log
MONTH_RE=$(printf "%02d|" $MONTHS); MONTH_RE="^(${MONTH_RE%|})$"

log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

busy_pct() {   # CPU busy percentage over 30 s from /proc/stat (idle + iowait count as free)
  read -r _ u1 n1 s1 i1 w1 x1 y1 _ < /proc/stat; sleep 30; read -r _ u2 n2 s2 i2 w2 x2 y2 _ < /proc/stat
  local t=$(( (u2+n2+s2+i2+w2+x2+y2) - (u1+n1+s1+i1+w1+x1+y1) )) f=$(( (i2+w2) - (i1+w1) ))
  echo $(( 100 - 100*f/t ))
}

complete() {   # every day of MONTHS present for every variable?
  local V n
  for V in $VARS; do
    n=$(ls "$L/temp/$WRUN"/UIB_${V}_????-??-??.nc 2>/dev/null | sed 's/.*_\([0-9]\{4\}\)-\([0-9]\{2\}\)-[0-9]\{2\}\.nc/\2/' | grep -cE "$MONTH_RE")
    [ "$n" -ge "$EXPECTED" ] || return 1
  done
  return 0
}

log "scheduler started: $WRUN $VARS months [$MONTHS], $NJOBS workers each, window $START-$STOP, busy limit $BUSY_MAX%"
while ! complete; do
  now=$(date +%s); target=$(date -d "today $START" +%s); [ "$target" -le "$now" ] && target=$(date -d "tomorrow $START" +%s)
  log "sleeping until $(date -d @$target '+%F %T')"
  sleep $(( target - now ))
  launched=0
  for try in $(seq 0 $RETRIES); do
    b=$(busy_pct)
    if [ "$b" -le "$BUSY_MAX" ]; then launched=1; break; fi
    log "try $((try+1)) of $((RETRIES+1)): mc4 CPU busy ${b}% > ${BUSY_MAX}%"
    [ "$try" -lt "$RETRIES" ] && sleep $(( RETRY_MIN*60 - 30 ))
  done
  if [ "$launched" = 0 ]; then
    log "still busy at the last try; skipping tonight"
    echo "$(date '+%F %T') $WRUN: night skipped, mc4 busy" >> "$CHAIN"
    continue
  fi
  # memory guard: never ask for more workers than the free memory can hold
  nvars=$(echo $VARS | wc -w); avail_gb=$(( $(awk '/MemAvailable/ {print $2}' /proc/meminfo) / 1048576 ))
  maxw=$(( avail_gb * MEM_FRACTION / 100 / MEM_PER_WORKER_GB / nvars )); nj=$NJOBS
  [ "$maxw" -lt "$nj" ] && nj=$maxw; [ "$nj" -lt 1 ] && nj=1
  log "launching $VARS, $nj workers each (requested $NJOBS; ${avail_gb} GB free, ${MEM_PER_WORKER_GB} GB/worker), nice 10, no new day after $STOP (CPU busy ${b}%)"
  echo "$(date '+%F %T') $WRUN: night window start, $nj workers each" >> "$CHAIN"
  for V in $VARS; do
    nice -n 10 $P run_var_remote.py --wrun $WRUN --var $V --syear 2011 --eyear 2020 --months $MONTHS --njobs $nj --stop-at $STOP >> "$L/${V,,}_$WRUN.log" 2>&1 &
  done
  wait
  for V in $VARS; do log "$V: $(grep -a '^finished' "$L/${V,,}_$WRUN.log" | tail -1)"; done
  echo "$(date '+%F %T') $WRUN: night window end" >> "$CHAIN"
done
log "all days present for $VARS; done"
echo "chain finished" >> "$CHAIN"
