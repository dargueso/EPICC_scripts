#!/usr/bin/env python
"""Quick check if zarr stores are consolidated."""
import os
import sys

def check_zarr_consolidated(zarr_path):
    """Check if a zarr store has consolidated metadata."""
    metadata_file = os.path.join(zarr_path, '.zmetadata')
    
    if os.path.exists(metadata_file):
        # Check file size to ensure it's not empty
        size = os.path.getsize(metadata_file)
        print(f"✓ {zarr_path}")
        print(f"  Consolidated: YES")
        print(f"  .zmetadata size: {size:,} bytes")
        return True
    else:
        print(f"✗ {zarr_path}")
        print(f"  Consolidated: NO")
        print(f"  Missing .zmetadata file")
        return False

if __name__ == "__main__":
    # Check both zarr stores
    PATH = '/home/dargueso/postprocessed/EPICC/EPICC_2km_ERA5_CMIP6anom'
    
    print("="*60)
    print("Checking Zarr Consolidation")
    print("="*60)
    
    zarr_files = [
        f"{PATH}/UIB_10MIN_RAIN.zarr",    
        f"{PATH}/UIB_01H_RAIN.zarr",
        f"{PATH}/UIB_DAY_RAIN.zarr",
    ]
    
    for zarr_path in zarr_files:
        print(f"\n{os.path.basename(zarr_path)}:")
        if os.path.exists(zarr_path):
            check_zarr_consolidated(zarr_path)
        else:
            print(f"  NOT FOUND: {zarr_path}")
    
    print("\n" + "="*60)
