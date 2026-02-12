#!/usr/bin/env python
"""
ATTRIBUTION ANALYSIS: Decompose changes in hourly rainfall extremes

This script answers the key question:
    "Can changes in hourly rainfall extremes be explained by changes in 
     daily totals alone, or is the temporal structure within days changing?"

CONCEPTUAL FRAMEWORK:
--------------------
We decompose the total change in hourly rainfall into two components:

1. DAILY TOTAL EFFECT (what we can explain):
   - How much would hourly extremes change if ONLY daily totals changed?
   - Assumption: temporal structure stays the same as present-day
   - Measured by: Synthetic Future - Present

2. STRUCTURAL CHANGE EFFECT (what's left unexplained):
   - How much additional change comes from evolving temporal structure?
   - Evidence that rainfall is becoming more/less concentrated within days
   - Measured by: Actual Future - Synthetic Future

3. TOTAL OBSERVED CHANGE:
   - What actually happens to hourly extremes
   - Measured by: Actual Future - Present
   - Should equal: Daily Total Effect + Structural Change Effect

INTERPRETATION:
--------------
- If structural effect ≈ 0: Changes explained by daily totals alone
- If structural effect > 0: Rainfall becoming MORE concentrated (higher Gini)
- If structural effect < 0: Rainfall becoming LESS concentrated (lower Gini)
- Large structural effects suggest evolving sub-daily dynamics

REQUIRED INPUTS:
---------------
1. percentiles_and_significance_{FREQ}_{test_type}_seqio.nc
   - Present and future hourly percentiles from observations
   
2. synthetic_future_{FREQ_HIGH}_from_{FREQ_LOW}_confidence.nc
   - Synthetic future: present structure + future daily totals
"""

import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

PATH_IN = '/home/dargueso/postprocessed/EPICC/'
PATH_OUT = '/home/dargueso/postprocessed/EPICC/'
WRUN_PRESENT = "EPICC_2km_ERA5"
WRUN_FUTURE = "EPICC_2km_ERA5_CMIP6anom"

# Frequency settings
FREQ_HIGH = '01H'  # High frequency (hourly)
FREQ_LOW = 'DAY'    # Low frequency (daily)

# Which percentiles to analyze in detail
# (should match what's in your percentiles file)
PERCENTILES_TO_ANALYZE = [50, 75, 90, 95, 99, 99.9]

# Thresholds for classification
FRACTION_THRESHOLD = 0.2  # Classify as "explained" if fraction > 0.8 or < -0.2

# Test type used in percentiles file
TEST_TYPE = 'mann_whitney'

# Bootstrap quantile to use from synthetic data (0.5 = median estimate)
BOOTSTRAP_QUANTILE = 0.5

test_suffix = "_test_100x100"

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def classify_attribution(fraction_daily):
    """
    Classify grid points based on what drives the change.
    
    Parameters:
    -----------
    fraction_daily : array
        Fraction of total change explained by daily totals
        
    Returns:
    --------
    classification : array of strings
        'daily_dominated': > 80% explained by daily totals
        'structural_dominated': < 20% explained by daily totals
        'mixed': between 20-80%
        'amplifying': total and daily effects have opposite signs
        'no_change': total change is negligible
    """
    classification = np.full(fraction_daily.shape, '', dtype='U20')
    
    # Where fraction is between 0.8 and 1.2: daily totals explain most
    classification[fraction_daily > 0.8] = 'daily_dominated'
    
    # Where fraction is between -0.2 and 0.2: structural change dominates
    classification[(fraction_daily >= -0.2) & (fraction_daily <= 0.2)] = 'structural_dominated'
    
    # Where fraction is between 0.2 and 0.8: mixed influence
    classification[(fraction_daily > 0.2) & (fraction_daily <= 0.8)] = 'mixed'
    
    # Where fraction < -0.2 or > 1.2: amplifying/dampening effects
    classification[(fraction_daily < -0.2) | (fraction_daily > 1.2)] = 'amplifying'
    
    return classification


def calculate_statistics(change_total, change_daily, change_structural):
    """
    Calculate summary statistics for the decomposition.
    
    Returns dictionary with:
    - Mean changes
    - Fraction of variance explained
    - Spatial correlation between components
    """
    stats_dict = {}
    
    # Mean changes (spatial average)
    stats_dict['mean_total_change'] = np.nanmean(change_total)
    stats_dict['mean_daily_effect'] = np.nanmean(change_daily)
    stats_dict['mean_structural_effect'] = np.nanmean(change_structural)
    
    # Variance decomposition (how much variance in each component)
    stats_dict['var_total'] = np.nanvar(change_total)
    stats_dict['var_daily'] = np.nanvar(change_daily)
    stats_dict['var_structural'] = np.nanvar(change_structural)
    
    # Spatial correlation between daily and structural effects
    valid_mask = ~(np.isnan(change_daily) | np.isnan(change_structural))
    if np.sum(valid_mask) > 10:
        corr = np.corrcoef(
            change_daily[valid_mask].flatten(),
            change_structural[valid_mask].flatten()
        )[0, 1]
        stats_dict['correlation_daily_structural'] = corr
    else:
        stats_dict['correlation_daily_structural'] = np.nan
    
    # Fraction of grid points where each effect dominates
    fraction_daily = change_daily / np.where(change_total != 0, change_total, np.nan)
    
    stats_dict['frac_daily_dominated'] = np.nanmean((fraction_daily > 0.8).astype(float))
    stats_dict['frac_structural_dominated'] = np.nanmean(
        ((fraction_daily >= -0.2) & (fraction_daily <= 0.2)).astype(float)
    )
    stats_dict['frac_mixed'] = np.nanmean(
        ((fraction_daily > 0.2) & (fraction_daily <= 0.8)).astype(float)
    )
    
    return stats_dict


def print_summary(percentile, stats_dict, units='mm/h'):
    """Print a summary of the decomposition for one percentile."""
    print(f"\n{'='*70}")
    print(f"PERCENTILE {percentile}th")
    print(f"{'='*70}")
    
    print(f"\nMean Changes (spatial average):")
    print(f"  Total observed change:        {stats_dict['mean_total_change']:+7.3f} {units}")
    print(f"  Daily total effect:           {stats_dict['mean_daily_effect']:+7.3f} {units}")
    print(f"  Structural change effect:     {stats_dict['mean_structural_effect']:+7.3f} {units}")
    
    # Calculate percentage contributions
    total_abs = abs(stats_dict['mean_total_change'])
    if total_abs > 0:
        daily_pct = 100 * stats_dict['mean_daily_effect'] / stats_dict['mean_total_change']
        struct_pct = 100 * stats_dict['mean_structural_effect'] / stats_dict['mean_total_change']
        print(f"\nContributions to total change:")
        print(f"  Daily totals explain:         {daily_pct:+6.1f}%")
        print(f"  Structural changes explain:   {struct_pct:+6.1f}%")
    
    print(f"\nSpatial variance:")
    print(f"  Total change variance:        {stats_dict['var_total']:.3e}")
    print(f"  Daily effect variance:        {stats_dict['var_daily']:.3e}")
    print(f"  Structural effect variance:   {stats_dict['var_structural']:.3e}")
    
    print(f"\nSpatial patterns:")
    print(f"  Correlation (daily vs struct): {stats_dict['correlation_daily_structural']:+.3f}")
    
    print(f"\nGrid point classification:")
    print(f"  Daily-dominated (>80%):       {100*stats_dict['frac_daily_dominated']:.1f}%")
    print(f"  Mixed influence (20-80%):     {100*stats_dict['frac_mixed']:.1f}%")
    print(f"  Structural-dominated (<20%):  {100*stats_dict['frac_structural_dominated']:.1f}%")


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def main():
    print("="*70)
    print("ATTRIBUTION DECOMPOSITION ANALYSIS")
    print("="*70)
    print(f"\nAnalyzing changes in {FREQ_HIGH} rainfall extremes")
    print(f"Attribution: Daily totals vs. temporal structure")
    
    # =========================================================================
    # STEP 1: LOAD DATA
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 1: Loading datasets")
    print("="*70)
    
    # Load observed percentiles (present and future)
    print(f"\n1a. Loading observed percentiles...")
    percentiles_file = (f'{PATH_IN}/{WRUN_PRESENT}/'
                       f'percentiles_and_significance_{FREQ_HIGH}_{TEST_TYPE}_seqio{test_suffix}.nc')
    
    if not os.path.exists(percentiles_file):
        raise FileNotFoundError(f"Cannot find: {percentiles_file}")
    
    ds_obs = xr.open_dataset(percentiles_file)
    print(f"    Loaded: {percentiles_file}")
    print(f"    Percentiles available: {ds_obs.percentile.values}")
    print(f"    Domain size: {len(ds_obs.y)} x {len(ds_obs.x)}")
    
    # Load synthetic future percentiles
    print(f"\n1b. Loading synthetic future percentiles...")
    synthetic_file = (f'{PATH_IN}/{WRUN_FUTURE}/'
                     f'synthetic_future_{FREQ_HIGH}_from_{FREQ_LOW}_confidence{test_suffix}.nc')
    
    if not os.path.exists(synthetic_file):
        raise FileNotFoundError(f"Cannot find: {synthetic_file}")
    
    ds_synth = xr.open_dataset(synthetic_file)
    print(f"    Loaded: {synthetic_file}")
    print(f"    Bootstrap quantiles: {ds_synth.bootstrap_quantile.values}")
    print(f"    Quantiles available: {ds_synth.coords['quantile'].values}")
    
    # =========================================================================
    # STEP 2: ALIGN DATA
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 2: Aligning datasets")
    print("="*70)
    
    # Extract the bootstrap quantile we want (typically 0.5 = median)
    print(f"\n2a. Extracting bootstrap quantile {BOOTSTRAP_QUANTILE}...")
    ds_synth_median = ds_synth.sel(bootstrap_quantile=BOOTSTRAP_QUANTILE, method='nearest')
    
    # Convert synthetic quantiles (0-1) to percentiles (0-100) for matching
    synthetic_percentiles = ds_synth_median.coords['quantile'].values * 100
    
    # Find matching percentiles
    print(f"\n2b. Finding matching percentiles...")
    available_percentiles = ds_obs.percentile.values
    
    # Match each requested percentile to closest available
    percentiles_matched = []
    for p in PERCENTILES_TO_ANALYZE:
        idx = np.argmin(np.abs(available_percentiles - p))
        percentiles_matched.append(available_percentiles[idx])
        if abs(available_percentiles[idx] - p) > 1.0:
            print(f"    Warning: Requested P{p} matched to P{available_percentiles[idx]}")
    
    percentiles_matched = np.unique(percentiles_matched)
    print(f"    Analyzing {len(percentiles_matched)} percentiles: {percentiles_matched}")
    
    # =========================================================================
    # STEP 3: CALCULATE DECOMPOSITION
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 3: Calculating decomposition")
    print("="*70)
    
    # Initialize output arrays
    n_percentiles = len(percentiles_matched)
    ny, nx = len(ds_obs.y), len(ds_obs.x)
    
    # Main decomposition components
    change_total = np.full((n_percentiles, ny, nx), np.nan, dtype=np.float32)
    change_daily = np.full((n_percentiles, ny, nx), np.nan, dtype=np.float32)
    change_structural = np.full((n_percentiles, ny, nx), np.nan, dtype=np.float32)
    
    # Attribution fractions
    fraction_daily = np.full((n_percentiles, ny, nx), np.nan, dtype=np.float32)
    fraction_structural = np.full((n_percentiles, ny, nx), np.nan, dtype=np.float32)
    
    # Percentage changes
    pct_change_total = np.full((n_percentiles, ny, nx), np.nan, dtype=np.float32)
    pct_change_daily = np.full((n_percentiles, ny, nx), np.nan, dtype=np.float32)
    pct_change_structural = np.full((n_percentiles, ny, nx), np.nan, dtype=np.float32)
    
    # Process each percentile
    print("\n3a. Computing decomposition for each percentile...")
    
    for i, perc in enumerate(percentiles_matched):
        print(f"    Processing P{perc}...")
        
        # Get observed data
        present = ds_obs.percentiles_present.sel(percentile=perc).values
        future = ds_obs.percentiles_future.sel(percentile=perc).values
        
        # Get synthetic data (match percentile value)
        # Find closest quantile in synthetic data
        target_quantile = perc / 100.0
        synth_idx = np.argmin(np.abs(ds_synth_median.coords['quantile'].values - target_quantile))
        synthetic = ds_synth_median.precipitation.isel(quantile=synth_idx).values
        
        # DECOMPOSITION:
        # Total change = Future - Present
        change_total[i] = future - present
        
        # Daily total effect = Synthetic - Present
        # (what would happen if only daily totals changed)
        change_daily[i] = synthetic - present
        
        # Structural change effect = Future - Synthetic
        # (the residual not explained by daily totals)
        change_structural[i] = future - synthetic
        
        # Calculate fractions (avoid division by zero)
        with np.errstate(divide='ignore', invalid='ignore'):
            fraction_daily[i] = change_daily[i] / change_total[i]
            fraction_structural[i] = change_structural[i] / change_total[i]
            
            # Percentage changes relative to present
            pct_change_total[i] = 100 * change_total[i] / present
            pct_change_daily[i] = 100 * change_daily[i] / present
            pct_change_structural[i] = 100 * change_structural[i] / present
    
    # =========================================================================
    # STEP 4: CLASSIFY GRID POINTS
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 4: Classifying attribution patterns")
    print("="*70)
    
    classification = np.full((n_percentiles, ny, nx), '', dtype='U20')
    
    for i, perc in enumerate(percentiles_matched):
        classification[i] = classify_attribution(fraction_daily[i])
        
        # Count classifications
        n_total = np.sum(~np.isnan(fraction_daily[i]))
        n_daily = np.sum(classification[i] == 'daily_dominated')
        n_struct = np.sum(classification[i] == 'structural_dominated')
        n_mixed = np.sum(classification[i] == 'mixed')
        n_amp = np.sum(classification[i] == 'amplifying')
        
        print(f"\n    P{perc} classification:")
        print(f"      Daily-dominated:      {n_daily:6d} ({100*n_daily/n_total:5.1f}%)")
        print(f"      Structural-dominated: {n_struct:6d} ({100*n_struct/n_total:5.1f}%)")
        print(f"      Mixed influence:      {n_mixed:6d} ({100*n_mixed/n_total:5.1f}%)")
        print(f"      Amplifying:           {n_amp:6d} ({100*n_amp/n_total:5.1f}%)")
    
    # =========================================================================
    # STEP 5: CALCULATE SUMMARY STATISTICS
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 5: Summary statistics")
    print("="*70)
    
    all_stats = {}
    for i, perc in enumerate(percentiles_matched):
        stats_dict = calculate_statistics(
            change_total[i],
            change_daily[i],
            change_structural[i]
        )
        all_stats[perc] = stats_dict
        print_summary(perc, stats_dict)
    
    # =========================================================================
    # STEP 6: SAVE RESULTS
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 6: Saving results")
    print("="*70)
    
    # Create output dataset
    ds_output = xr.Dataset(
        data_vars={
            # Absolute changes
            'change_total': (['percentile', 'y', 'x'], change_total,
                            {'long_name': 'Total observed change (Future - Present)',
                             'units': f'mm/{FREQ_HIGH}',
                             'description': 'Actual change in hourly rainfall percentiles'}),
            
            'change_daily_effect': (['percentile', 'y', 'x'], change_daily,
                                   {'long_name': 'Daily total effect (Synthetic - Present)',
                                    'units': f'mm/{FREQ_HIGH}',
                                    'description': 'Change if only daily totals changed (structure constant)'}),
            
            'change_structural_effect': (['percentile', 'y', 'x'], change_structural,
                                        {'long_name': 'Structural change effect (Future - Synthetic)',
                                         'units': f'mm/{FREQ_HIGH}',
                                         'description': 'Residual change from evolving temporal structure'}),
            
            # Fractions
            'fraction_daily': (['percentile', 'y', 'x'], fraction_daily,
                             {'long_name': 'Fraction explained by daily totals',
                              'units': 'dimensionless',
                              'description': 'Daily effect / Total change (values >1 or <0 indicate amplification)'}),
            
            'fraction_structural': (['percentile', 'y', 'x'], fraction_structural,
                                  {'long_name': 'Fraction explained by structure',
                                   'units': 'dimensionless',
                                   'description': 'Structural effect / Total change'}),
            
            # Percentage changes
            'pct_change_total': (['percentile', 'y', 'x'], pct_change_total,
                               {'long_name': 'Total percent change',
                                'units': '%',
                                'description': '100 * (Future - Present) / Present'}),
            
            'pct_change_daily': (['percentile', 'y', 'x'], pct_change_daily,
                                {'long_name': 'Daily effect percent change',
                                 'units': '%',
                                 'description': '100 * (Synthetic - Present) / Present'}),
            
            'pct_change_structural': (['percentile', 'y', 'x'], pct_change_structural,
                                     {'long_name': 'Structural effect percent change',
                                      'units': '%',
                                      'description': '100 * (Future - Synthetic) / Present'}),
            
            # Classification
            'classification': (['percentile', 'y', 'x'], classification,
                             {'long_name': 'Attribution classification',
                              'description': 'daily_dominated, structural_dominated, mixed, or amplifying'}),
            
            # Reference fields
            'present': (['percentile', 'y', 'x'], 
                       ds_obs.percentiles_present.sel(percentile=percentiles_matched).values,
                       {'long_name': 'Present climate percentiles',
                        'units': f'mm/{FREQ_HIGH}'}),
            
            'future': (['percentile', 'y', 'x'],
                      ds_obs.percentiles_future.sel(percentile=percentiles_matched).values,
                      {'long_name': 'Future climate percentiles',
                       'units': f'mm/{FREQ_HIGH}'}),
            
            'lat': (['y', 'x'], ds_obs.lat.values),
            'lon': (['y', 'x'], ds_obs.lon.values),
        },
        coords={
            'percentile': percentiles_matched,
            'y': ds_obs.y.values,
            'x': ds_obs.x.values,
        },
        attrs={
            'title': 'Attribution decomposition of hourly rainfall changes',
            'description': (
                'Decomposition of changes in hourly rainfall extremes into: '
                '(1) Daily total effect - change from daily rainfall amounts assuming constant structure, '
                '(2) Structural effect - change from evolving temporal distribution within days'
            ),
            'methodology': (
                'Total change = Future - Present, '
                'Daily effect = Synthetic - Present, '
                'Structural effect = Future - Synthetic'
            ),
            'freq_high': FREQ_HIGH,
            'freq_low': FREQ_LOW,
            'bootstrap_quantile_used': BOOTSTRAP_QUANTILE,
            'present_period': WRUN_PRESENT,
            'future_period': WRUN_FUTURE,
            'interpretation': (
                'Positive structural effect: rainfall becoming more concentrated within days. '
                'Negative structural effect: rainfall becoming more evenly distributed. '
                'Fraction_daily > 0.8: changes explained by daily totals alone. '
                'Fraction_daily < 0.2: structural changes dominate.'
            ),
            'note': 'This is a counterfactual analysis testing if structure changes matter'
        }
    )
    
    # Add summary statistics as global attributes
    for perc, stats_dict in all_stats.items():
        for key, value in stats_dict.items():
            attr_name = f'P{int(perc)}_{key}'
            ds_output.attrs[attr_name] = float(value) if not np.isnan(value) else 'nan'
    
    # Save
    output_file = (f'{PATH_OUT}/{WRUN_PRESENT}/'
                  f'attribution_decomposition_{FREQ_HIGH}{test_suffix}.nc')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    print(f"\nSaving to: {output_file}")
    ds_output.to_netcdf(output_file)
    print(f"Saved successfully!")
    
    # =========================================================================
    # STEP 7: FINAL SUMMARY
    # =========================================================================
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    
    print("\nKey Findings:")
    print("-" * 70)
    
    for perc in percentiles_matched:
        stats = all_stats[perc]
        
        total_chg = stats['mean_total_change']
        daily_chg = stats['mean_daily_effect']
        struct_chg = stats['mean_structural_effect']
        
        if abs(total_chg) > 0.01:  # Only report if meaningful change
            daily_pct = 100 * daily_chg / total_chg
            struct_pct = 100 * struct_chg / total_chg
            
            print(f"\nP{int(perc):02d}: Total change = {total_chg:+.3f} mm/{FREQ_HIGH}")
            print(f"     Daily totals contribute:   {daily_pct:+6.1f}% ({daily_chg:+.3f} mm/{FREQ_HIGH})")
            print(f"     Structural changes add:    {struct_pct:+6.1f}% ({struct_chg:+.3f} mm/{FREQ_HIGH})")
            
            if abs(struct_pct) > 30:
                print(f"     --> STRUCTURAL CHANGES ARE IMPORTANT!")
            elif abs(struct_pct) < 10:
                print(f"     --> Changes well-explained by daily totals alone")
    
    print("\n" + "="*70)
    print(f"Output saved to: {output_file}")
    print("="*70)
    
    # Cleanup
    ds_obs.close()
    ds_synth.close()
    ds_output.close()


if __name__ == "__main__":
    main()
