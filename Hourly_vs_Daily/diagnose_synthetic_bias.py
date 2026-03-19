#!/usr/bin/env python
"""
Diagnostic: compare present observed hourly vs. synthetic present hourly
at the center pixel of the 21x21 test area.

Checks:
  1. Quantile-quantile comparison (center pixel vs center pixel)
  2. Mass conservation (total synthetic rain == total daily input rain)
  3. Mean N_wet per wet day (observed vs. synthetic)
  4. Per-daily-bin mean hourly intensity comparison
"""
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["font.size"] = 12

# =============================================================================
# CONFIG
# =============================================================================
PATH_IN = '/home/dargueso/postprocessed/EPICC/'
PATH_OUT = '/home/dargueso/Analyses/EPICC/Hourly_vs_Daily/'

WRUN_PRESENT = 'EPICC_2km_ERA5'
test_suffix  = '_test_21x21'

WET_VALUE_HIGH = 0.1   # mm - wet hourly threshold
WET_VALUE_LOW  = 1.0   # mm - wet day threshold

CENTER = 10             # center index in 21x21 grid

QUANTILES = np.array([0.10, 0.20, 0.25, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80,
                      0.85, 0.90, 0.95, 0.98, 0.99, 0.995, 0.999], dtype=np.float64)

# Bins for daily total (must match condprob bins)
BINS_LOW  = np.append(np.arange(0, 100, 5), np.inf)

# =============================================================================
# LOAD DATA
# =============================================================================
print("Loading observed hourly data...")
zarr_path = f'{PATH_IN}/{WRUN_PRESENT}/UIB_01H_RAIN{test_suffix}.zarr'
try:
    ds = xr.open_zarr(zarr_path, consolidated=True)
except KeyError:
    ds = xr.open_zarr(zarr_path, consolidated=False)

rain_all = ds.RAIN.astype(np.float32)                      # (time, 21, 21)
rain_ctr = rain_all.isel(y=CENTER, x=CENTER).values        # (time,) center pixel

print("Loading synthetic present...")
syn_file = f'{PATH_IN}/{WRUN_PRESENT}/synthetic_future_01H_from_DAY_confidence{test_suffix}.nc'
ds_syn = xr.open_dataset(syn_file).squeeze()
ds_syn_q = ds_syn.isel(quantile=slice(10, -2))             # same slice used in plot
qtiles = ds_syn_q['quantile'].values                       # quantile axis used in plot

# Synthetic median at center pixel
syn_med_h  = ds_syn_q.sel(bootstrap_quantile=0.5).hourly_intensity.isel(y=CENTER, x=CENTER).values
syn_lo_h   = ds_syn_q.sel(bootstrap_quantile=0.025).hourly_intensity.isel(y=CENTER, x=CENTER).values
syn_hi_h   = ds_syn_q.sel(bootstrap_quantile=0.975).hourly_intensity.isel(y=CENTER, x=CENTER).values

syn_med_dm = ds_syn_q.sel(bootstrap_quantile=0.5).max_hourly_intensity.isel(y=CENTER, x=CENTER).values
syn_lo_dm  = ds_syn_q.sel(bootstrap_quantile=0.025).max_hourly_intensity.isel(y=CENTER, x=CENTER).values
syn_hi_dm  = ds_syn_q.sel(bootstrap_quantile=0.975).max_hourly_intensity.isel(y=CENTER, x=CENTER).values

print("Loading daily data for mass check...")
zarr_day = f'{PATH_IN}/{WRUN_PRESENT}/UIB_DAY_RAIN{test_suffix}.zarr'
try:
    ds_day = xr.open_zarr(zarr_day, consolidated=True)
except KeyError:
    ds_day = xr.open_zarr(zarr_day, consolidated=False)

rain_day_ctr = ds_day.RAIN.isel(y=CENTER, x=CENTER).values   # daily totals, center pixel

# =============================================================================
# DIAGNOSTIC 1 — CENTER-PIXEL vs CENTER-PIXEL quantile comparison
# =============================================================================
print("\n--- Diagnostic 1: Center-pixel observed quantiles ---")

# Center pixel observed
rain_ctr_wet = rain_ctr[rain_ctr > WET_VALUE_HIGH]
obs_ctr_qtiles = np.quantile(rain_ctr_wet, QUANTILES)

# All-21x21 observed
rain_flat = rain_all.values.reshape(rain_all.shape[0], -1)   # (time, 441)
rain_flat_wet = rain_flat[rain_flat > WET_VALUE_HIGH]
obs_all_qtiles = np.quantile(rain_flat_wet, QUANTILES)

print(f"  N wet hours - center pixel: {len(rain_ctr_wet)}")
print(f"  N wet hours - all 21x21:    {len(rain_flat_wet)}")
print(f"  Mean wet-hour intensity - center: {rain_ctr_wet.mean():.3f} mm")
print(f"  Mean wet-hour intensity - all:    {rain_flat_wet.mean():.3f} mm")

# Compare synthetic vs center pixel observed
print("\n  Quantile comparison (center pixel obs vs synthetic median):")
print(f"  {'Quantile':>10} {'Obs_ctr':>10} {'Syn_med':>10} {'Ratio':>8}")
for i, q in enumerate(QUANTILES):
    if q in qtiles:
        j = np.searchsorted(qtiles, q)
        if j < len(qtiles) and np.isclose(qtiles[j], q):
            print(f"  {q:>10.3f} {obs_ctr_qtiles[i]:>10.3f} {syn_med_h[j]:>10.3f} {syn_med_h[j]/obs_ctr_qtiles[i]:>8.2f}")

# =============================================================================
# DIAGNOSTIC 2 — Mass conservation
# =============================================================================
print("\n--- Diagnostic 2: Mass conservation ---")

# Total observed daily rain at center pixel (only wet days)
wet_days_mask = rain_day_ctr >= WET_VALUE_LOW
total_observed_daily = rain_day_ctr[wet_days_mask].sum()

# Total observed hourly rain at center pixel (all hours on wet days)
# Need to align hourly with daily
n_days = len(rain_day_ctr)
n_hours = len(rain_ctr)
# Trim hourly to complete days
n_complete_days = n_hours // 24
rain_ctr_daily_blocks = rain_ctr[:n_complete_days * 24].reshape(n_complete_days, 24)
rain_day_trim = rain_day_ctr[:n_complete_days]
wet_days_trim = rain_day_trim >= WET_VALUE_LOW

total_obs_hourly_on_wet_days = rain_ctr_daily_blocks[wet_days_trim, :].sum()
print(f"  Total observed daily (wet days): {total_observed_daily:.1f} mm")
print(f"  Total observed hourly (wet days): {total_obs_hourly_on_wet_days:.1f} mm")
print(f"  Discrepancy: {abs(total_observed_daily - total_obs_hourly_on_wet_days):.2f} mm ({abs(total_observed_daily - total_obs_hourly_on_wet_days)/total_observed_daily*100:.2f}%)")

# =============================================================================
# DIAGNOSTIC 3 — N_wet per wet day
# =============================================================================
print("\n--- Diagnostic 3: N_wet distribution ---")

n_wet_obs = (rain_ctr_daily_blocks[wet_days_trim, :] > WET_VALUE_HIGH).sum(axis=1)
print(f"  Observed N wet hours per wet day:")
print(f"    Mean: {n_wet_obs.mean():.2f}")
print(f"    Std:  {n_wet_obs.std():.2f}")
print(f"    Median: {np.median(n_wet_obs):.1f}")
print(f"    P90: {np.percentile(n_wet_obs, 90):.1f}")
print(f"    Distribution (1-6 hours, %, then >= 7):")
for n in range(1, 7):
    pct = (n_wet_obs == n).mean() * 100
    print(f"      n_wet={n}: {pct:.1f}%")
pct_7plus = (n_wet_obs >= 7).mean() * 100
print(f"      n_wet>=7: {pct_7plus:.1f}%")

# =============================================================================
# DIAGNOSTIC 4 — Per daily-bin mean wet-hour intensity
# =============================================================================
print("\n--- Diagnostic 4: Per daily-bin mean wet-hour intensity ---")

bin_labels = [f"{int(BINS_LOW[i])}-{int(BINS_LOW[i+1]) if not np.isinf(BINS_LOW[i+1]) else 'inf'}"
              for i in range(len(BINS_LOW)-1)]

print(f"  {'Bin (mm/day)':>15} {'N_days':>8} {'Mean_n_wet':>12} {'Mean_h_intens':>15}")
for j in range(len(BINS_LOW)-1):
    lo, hi = BINS_LOW[j], BINS_LOW[j+1]
    mask = (rain_day_trim >= lo) & (rain_day_trim < hi) & wet_days_trim
    if mask.sum() == 0:
        continue
    n_days_bin = mask.sum()
    rain_hours_bin = rain_ctr_daily_blocks[mask, :]
    n_wet_bin = (rain_hours_bin > WET_VALUE_HIGH).sum(axis=1).mean()
    wet_hours_bin = rain_hours_bin[rain_hours_bin > WET_VALUE_HIGH]
    mean_intens_bin = wet_hours_bin.mean() if len(wet_hours_bin) > 0 else np.nan
    print(f"  {bin_labels[j]:>15} {n_days_bin:>8} {n_wet_bin:>12.2f} {mean_intens_bin:>15.3f}")

# =============================================================================
# DIAGNOSTIC 5 — Load condprob and inspect center pixel distributions
# =============================================================================
print("\n--- Diagnostic 5: Condprob at center pixel ---")
condprob_file = f'{PATH_IN}/{WRUN_PRESENT}/condprob_01H_given_DAY{test_suffix}.nc'
try:
    ds_cp = xr.open_dataset(condprob_file)
    cp_ctr = ds_cp.isel(y=CENTER, x=CENTER)

    # n_events per daily bin at center pixel
    n_ev = cp_ctr.n_events.values
    total_events = n_ev.sum()
    print(f"  Total wet-day events in condprob: {total_events}")
    print(f"  N_wet events per bin (first 10 bins):")
    for j in range(min(10, len(n_ev))):
        print(f"    bin {j} ({bin_labels[j]}): {n_ev[j]}")

    # Mean of n_wet distribution at center pixel for bin 1 (5-10mm days)
    bin_idx = 1  # 5-10mm bin
    n_wet_dist = cp_ctr.cond_prob_n_wet.isel(**{'bin_DAY': bin_idx}).values
    n_wet_vals = ds_cp.n_wet_timesteps.values
    mean_n_wet_dist = (n_wet_dist * n_wet_vals).sum()
    print(f"\n  Mean N_wet from condprob for 5-10mm days: {mean_n_wet_dist:.2f}")

    # Compute from observed for 5-10mm days
    mask_5_10 = (rain_day_trim >= 5) & (rain_day_trim < 10) & wet_days_trim
    if mask_5_10.sum() > 0:
        n_wet_obs_5_10 = (rain_ctr_daily_blocks[mask_5_10, :] > WET_VALUE_HIGH).sum(axis=1).mean()
        print(f"  Mean N_wet from obs for 5-10mm days: {n_wet_obs_5_10:.2f}")

    ds_cp.close()
except FileNotFoundError:
    print(f"  Condprob file not found: {condprob_file}")

# =============================================================================
# PLOTS
# =============================================================================
print("\nGenerating diagnostic plots...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Observed center vs all-21x21 vs synthetic
ax = axes[0]
ax.plot(QUANTILES, obs_ctr_qtiles, label='Obs center pixel', color='#2E86AB',
        linewidth=2, marker='o', markersize=4)
ax.plot(QUANTILES, obs_all_qtiles, label='Obs 21×21 pooled', color='#2E86AB',
        linewidth=1.5, marker='^', markersize=4, linestyle='--', alpha=0.7)

# Overlay synthetic where quantiles match
ax.plot(qtiles, syn_med_h, label='Synthetic median', color='#F18F01',
        linewidth=2, linestyle='-')
ax.fill_between(qtiles, syn_lo_h, syn_hi_h, color='#F18F01', alpha=0.2,
                label='Synthetic CI (2.5–97.5%)')

ax.set_yscale('log')
ax.set_xlabel('Quantile', fontweight='bold')
ax.set_ylabel('1-hour precipitation (mm)', fontweight='bold')
ax.set_title('Hourly Intensity: Center-pixel vs Synthetic')
ax.legend(fontsize=9)
ax.grid(True, linestyle=':', alpha=0.6)

# Plot 2: Daily-max quantiles
ax2 = axes[1]
pres_dmax     = rain_all.resample(time='1D').max()
pres_dtot     = rain_all.resample(time='1D').sum()
pres_dmax_ctr = pres_dmax.isel(y=CENTER, x=CENTER).values
pres_dtot_ctr = pres_dtot.isel(y=CENTER, x=CENTER).values

dmax_wet_ctr = pres_dmax_ctr[pres_dtot_ctr >= WET_VALUE_LOW]
obs_dmax_ctr_q = np.quantile(dmax_wet_ctr, QUANTILES)

ax2.plot(QUANTILES, obs_dmax_ctr_q, label='Obs center pixel', color='#2E86AB',
         linewidth=2, marker='o', markersize=4)
ax2.plot(qtiles, syn_med_dm, label='Synthetic median', color='#F18F01', linewidth=2)
ax2.fill_between(qtiles, syn_lo_dm, syn_hi_dm, color='#F18F01', alpha=0.2,
                 label='Synthetic CI (2.5–97.5%)')
ax2.set_yscale('log')
ax2.set_xlabel('Quantile', fontweight='bold')
ax2.set_ylabel('Daily-max 1-hour precipitation (mm)', fontweight='bold')
ax2.set_title('Daily-Max Hourly: Center-pixel vs Synthetic')
ax2.legend(fontsize=9)
ax2.grid(True, linestyle=':', alpha=0.6)

fig.tight_layout()
outfile = f'{PATH_OUT}/diagnostic_synthetic_bias.png'
fig.savefig(outfile, dpi=150, bbox_inches='tight', facecolor='white')
print(f"Saved: {outfile}")
plt.close(fig)

print("\nDone.")
