#!/usr/bin/env python
"""
Diagnostic script to identify problematic tiles before processing.
Checks for common data issues that cause failures.
"""

import xarray as xr
import numpy as np
import epicc_config as cfg
from glob import glob
import os
from itertools import product
import warnings


def check_tile_files(ny, nx, fq, wrun):
    """
    Check if files for a tile exist and are valid.
    
    Returns:
    --------
    dict with diagnostic information
    """
    tile_id = f"{ny}y-{nx}x"
    diagnostics = {
        'tile_id': tile_id,
        'present_files_exist': False,
        'future_files_exist': False,
        'file_count_match': False,
        'present_loadable': False,
        'future_loadable': False,
        'time_conflict': False,
        'present_has_rain': False,
        'future_has_rain': False,
        'time_steps_present': 0,
        'time_steps_future': 0,
        'errors': []
    }
    
    # Check file existence
    filespath_p = f'{cfg.path_in}/{wrun}/split_files_tiles_50/{cfg.patt_in}_{fq}_RAIN_20??-??'
    filespath_f = filespath_p.replace('EPICC_2km_ERA5', 'EPICC_2km_ERA5_CMIP6anom')
    filesin_p = sorted(glob(f'{filespath_p}_{ny}y-{nx}x.nc'))
    filesin_f = sorted(glob(f'{filespath_f}_{ny}y-{nx}x.nc'))
    
    diagnostics['present_files_exist'] = len(filesin_p) > 0
    diagnostics['future_files_exist'] = len(filesin_f) > 0
    diagnostics['file_count_match'] = len(filesin_p) == len(filesin_f)
    
    if not diagnostics['present_files_exist']:
        diagnostics['errors'].append(f"No present files found")
        return diagnostics
    
    if not diagnostics['future_files_exist']:
        diagnostics['errors'].append(f"No future files found")
        return diagnostics
    
    if not diagnostics['file_count_match']:
        diagnostics['errors'].append(f"File count mismatch: {len(filesin_p)} present vs {len(filesin_f)} future")
    
    # Try loading present scenario
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            ds_p = xr.open_mfdataset(
                filesin_p,
                concat_dim="time",
                combine="nested",
                drop_variables=['time_bnds']
            )
            diagnostics['present_loadable'] = True
            diagnostics['present_has_rain'] = 'RAIN' in ds_p
            if 'time' in ds_p.dims:
                diagnostics['time_steps_present'] = len(ds_p.time)
            ds_p.close()
    except Exception as e:
        diagnostics['errors'].append(f"Present loading error: {str(e)}")
        if 'time_bnds' in str(e) or 'time' in str(e):
            diagnostics['time_conflict'] = True
    
    # Try loading future scenario
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            ds_f = xr.open_mfdataset(
                filesin_f,
                concat_dim="time",
                combine="nested",
                drop_variables=['time_bnds']
            )
            diagnostics['future_loadable'] = True
            diagnostics['future_has_rain'] = 'RAIN' in ds_f
            if 'time' in ds_f.dims:
                diagnostics['time_steps_future'] = len(ds_f.time)
            ds_f.close()
    except Exception as e:
        diagnostics['errors'].append(f"Future loading error: {str(e)}")
        if 'time_bnds' in str(e) or 'time' in str(e):
            diagnostics['time_conflict'] = True
    
    return diagnostics


def diagnose_all_tiles(fq='01H', wrun='EPICC_2km_ERA5', tile_size=50):
    """
    Check all tiles and report problems.
    """
    print("="*80)
    print(f"TILE DIAGNOSTICS: {wrun}, {fq}")
    print("="*80)
    
    # Get tile list
    filesin = sorted(glob(f'{cfg.path_in}/{wrun}/RAIN/{cfg.patt_in}_{fq}_RAIN_20??-??.nc'))
    if not filesin:
        print("ERROR: No reference files found!")
        return
    
    files_ref = xr.open_dataset(filesin[0])
    nlats = files_ref.sizes['y']
    nlons = files_ref.sizes['x']
    files_ref.close()
    
    lonsteps = [f'{nn:03d}' for nn in range(nlons//tile_size+1)]
    latsteps = [f'{nn:03d}' for nn in range(nlats//tile_size+1)]
    xytiles = list(product(latsteps, lonsteps))
    
    print(f"\nTotal tiles to check: {len(xytiles)}")
    print("-"*80)
    
    # Diagnostic counters
    issues = {
        'no_present_files': [],
        'no_future_files': [],
        'file_count_mismatch': [],
        'present_not_loadable': [],
        'future_not_loadable': [],
        'time_conflict': [],
        'no_rain_var': [],
        'time_mismatch': []
    }
    
    all_good = []
    
    # Check each tile
    for ny, nx in xytiles:
        diag = check_tile_files(ny, nx, fq, wrun)
        
        # Categorize issues
        if not diag['present_files_exist']:
            issues['no_present_files'].append(diag['tile_id'])
        if not diag['future_files_exist']:
            issues['no_future_files'].append(diag['tile_id'])
        if diag['present_files_exist'] and diag['future_files_exist'] and not diag['file_count_match']:
            issues['file_count_mismatch'].append(diag['tile_id'])
        if diag['present_files_exist'] and not diag['present_loadable']:
            issues['present_not_loadable'].append(diag['tile_id'])
        if diag['future_files_exist'] and not diag['future_loadable']:
            issues['future_not_loadable'].append(diag['tile_id'])
        if diag['time_conflict']:
            issues['time_conflict'].append(diag['tile_id'])
        if (diag['present_loadable'] and not diag['present_has_rain']) or \
           (diag['future_loadable'] and not diag['future_has_rain']):
            issues['no_rain_var'].append(diag['tile_id'])
        if diag['present_loadable'] and diag['future_loadable'] and \
           diag['time_steps_present'] != diag['time_steps_future']:
            issues['time_mismatch'].append(f"{diag['tile_id']} (P:{diag['time_steps_present']} F:{diag['time_steps_future']})")
        
        # Track good tiles
        if diag['present_loadable'] and diag['future_loadable'] and \
           diag['present_has_rain'] and diag['future_has_rain'] and \
           not diag['errors']:
            all_good.append(diag['tile_id'])
    
    # Report findings
    print("\n" + "="*80)
    print("DIAGNOSTICS SUMMARY")
    print("="*80)
    
    total_issues = sum(len(v) for v in issues.values() if isinstance(v, list))
    
    if total_issues == 0:
        print("\n✓ ALL TILES LOOK GOOD!")
        print(f"  {len(all_good)} tiles ready for processing")
    else:
        print(f"\n⚠ FOUND {total_issues} POTENTIAL ISSUES\n")
        
        for issue_type, tiles in issues.items():
            if tiles:
                print(f"\n{issue_type.replace('_', ' ').title()}:")
                print(f"  Count: {len(tiles)}")
                if len(tiles) <= 10:
                    for tile in tiles:
                        print(f"    - {tile}")
                else:
                    for tile in tiles[:5]:
                        print(f"    - {tile}")
                    print(f"    ... and {len(tiles)-5} more")
        
        print(f"\n✓ Tiles without issues: {len(all_good)}")
    
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    if issues['time_conflict']:
        print("\n• Time dimension conflicts detected:")
        print("  The robust version handles this automatically by dropping 'time_bnds'")
        print("  Use: create_multiple_significance_percentiles_robust.py")
    
    if issues['present_not_loadable'] or issues['future_not_loadable']:
        print("\n• Some files cannot be loaded (NetCDF errors):")
        print("  These tiles may have corrupted files")
        print("  Check the specific error messages in the log")
        print("  Consider regenerating these tiles if possible")
    
    if issues['file_count_mismatch']:
        print("\n• File count mismatches detected:")
        print("  Present and future scenarios have different numbers of files")
        print("  This will cause processing to fail")
    
    if issues['time_mismatch']:
        print("\n• Time step count mismatches detected:")
        print("  Present and future scenarios have different time lengths")
        print("  This may indicate incomplete data")
    
    # Save detailed report
    report_file = f'tile_diagnostics_{fq}.txt'
    with open(report_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"TILE DIAGNOSTICS: {wrun}, {fq}\n")
        f.write("="*80 + "\n\n")
        
        f.write("TILES WITH ISSUES:\n")
        f.write("-"*80 + "\n")
        for issue_type, tiles in issues.items():
            if tiles:
                f.write(f"\n{issue_type.replace('_', ' ').title()}:\n")
                for tile in tiles:
                    f.write(f"  {tile}\n")
        
        f.write("\n\nTILES READY FOR PROCESSING:\n")
        f.write("-"*80 + "\n")
        for tile in all_good:
            f.write(f"  {tile}\n")
    
    print(f"\n✓ Detailed report saved to: {report_file}")


if __name__ == "__main__":
    import sys
    
    # Allow command line arguments
    fq = sys.argv[1] if len(sys.argv) > 1 else '01H'
    
    diagnose_all_tiles(fq=fq)
