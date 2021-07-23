#!/usr/bin/env python
"""
#####################################################################
# Author: Daniel Argueso <daniel>
# Date:   2021-06-25T18:11:07+02:00
# Email:  d.argueso@uib.es
# Last modified by:   daniel
# Last modified time: 2021-06-25T18:11:08+02:00
#
# @Project@
# Version: x.0 (Beta)
# Description:
#
# Dependencies:
#
# Files:
#
#####################################################################
"""



import xarray
import dask
import numpy
import epicc_config as cfg


from dask.array.percentile import merge_percentiles
import numbers

wrun = cfg.wrf_runs[0]
qtiles = numpy.asarray(cfg.qtiles)
###########################################################
###########################################################

# for year in $(seq -w 2011 2020); do ncrcat -v RAIN UIB_10MIN_RAIN_${year}-??.nc merged/UIB_10MIN_RAIN_${year}.nc; done
# for file in $(ls UIB_01H_RAIN_*.nc);do ncks --cnk_dmn y,10 --cnk_dmn x,10 ${file} chunked/${file}; done
def main():

    # Check initial time
    ctime0=checkpoint(0)
    #Open the data
    # Open the data in xarray
    #ds = xarray.open_mfdataset(f'{cfg.path_in}/{wrun}/chunked/{cfg.patt_in}_01H_RAIN_201[1-4]-??.nc',\
    #    combine='nested', concat_dim='time', chunks={'y': 10, 'x': 10})

    ds = xarray.open_mfdataset(f'{cfg.path_in}/{wrun}/chunked/{cfg.patt_in}_01H_RAIN_2011-??.nc',\
        combine='nested', concat_dim='time', chunks={'y': 10, 'x': 10})

    print ("Loading chunked data")
    ctime1=checkpoint(ctime0)

    # Dask has a parallel percentile algorithm, but it only works on 1d data
    dask.array.percentile(ds.RAIN[:,0,0].data, 50).compute()

    print ("Calculating percentiles 1D dask")
    ctime2=checkpoint(ctime1)

    # Calculate multiple percentiles at once

    pcts = approx_percentile(ds.RAIN, qtiles*100., 'time')

    print ("Calculating percentiles ND dask")
    ctime3=checkpoint(ctime2)

    pcts.to_netcdf(f'{cfg.path_in}/{wrun}/{cfg.patt_in}_01H_RAIN_2011-2020-qtiles_dask.nc')

    print ("Saving pct file dask")
    ctime4=checkpoint(ctime3)



    ds_nocnk = xarray.open_mfdataset(f'{cfg.path_in}/{wrun}/{cfg.patt_in}_01H_RAIN_2011-??.nc',combine='nested', concat_dim='time')
    print ("Loading regular data")
    ctime5=checkpoint(ctime4)


    pcts2 = ds.RAIN.load().quantile(qtiles,dim=['time'])

    print ("Calculating percentiles  xarray")
    ctime6=checkpoint(ctime5)



    pcts2.to_netcdf(f'{cfg.path_in}/{wrun}/{cfg.patt_in}_01H_RAIN_2011-2020-qtiles_xarray.nc')
    print ("Saving pct file xarray")
    ctime7=checkpoint(ctime6)

    import pdb; pdb.set_trace()

# Extend Dask's algorithm to Nd datasets

def merge_approx_percentile(chunk_pcts, chunk_counts, finalpcts, pcts, axis):
    """
    Merge percentile blocks together

    A Nd implementation of dask.array.percentile.merge_percentiles
    """
    # First axis of chunk_pcts is the pct values
    assert chunk_pcts.shape[0] == len(pcts)
    # Remainder are the chunk results, stacked along 'axis'
    assert chunk_pcts.shape[1:] == chunk_counts.shape

    # Do a manual apply along axis, using the size of chunk_counts as it has the original dimensions
    Ni, Nk = chunk_counts.shape[:axis], chunk_counts.shape[axis+1:]

    # Output array has the values for each pct
    out = numpy.empty((len(finalpcts), *Ni, *Nk), dtype=chunk_pcts.dtype)

    # We have the same percentiles for each chunk
    pcts = numpy.tile(pcts, (chunk_counts.shape[axis],1))

    # Loop over the non-axis dimensions of the original array
    for ii in numpy.ndindex(Ni):
        for kk in numpy.ndindex(Nk):
            # Use dask's merge_percentiles
            out[ii + numpy.s_[...,] + kk] = merge_percentiles(finalpcts, pcts, chunk_pcts[numpy.s_[:,] +  ii + numpy.s_[:,] + kk].T, Ns=chunk_counts[ii + numpy.s_[:,] + kk])

    return out


def dask_approx_percentile(array: dask.array.array, pcts, axis: int):
    """
    Get the approximate percentiles of a Dask dataset along 'axis'

    Args:
        array: Dask Nd array
        pcts: List of percentiles to calculate, within the interval [0,100]
        axis: Axis to reduce

    Returns:
        Dask array with first axis the percentiles from 'pcts', remaining axes from
        'array' reduced along 'axis'
    """
    if isinstance(pcts, numbers.Number):
        pcts = [pcts]

    # The chunk sizes with each chunk reduced along 'axis'
    chunks = list(array.chunks)
    chunks[axis] = [1 for c in chunks[axis]]

    # Reproduce behaviour of dask.array.percentile, adding in '0' and '100' percentiles
    finalpcts = pcts.copy()
    pcts = numpy.pad(pcts, 1, mode="constant")
    pcts[-1] = 100

    # Add the percentile size to the start of 'chunks'
    chunks.insert(0, len(pcts))

    # The percentile of each chunk along 'axis'
    chunk_pcts = dask.array.map_blocks(numpy.nanpercentile, array, pcts, axis, keepdims=True, chunks=chunks, meta=numpy.array((), dtype=array.dtype))
    # The count of each chunk along 'axis'
    chunk_counts = dask.array.map_blocks(numpy.ma.count, array, axis, keepdims=True, chunks=chunks[1:], meta=numpy.array((), dtype='int64'))

    # Now change the chunk size to the final size
    chunks[0] = len(finalpcts)
    chunks.pop(axis+1)
    # Merge the chunks together using Dask's merge_percentiles function
    merged_pcts = dask.array.map_blocks(merge_approx_percentile, chunk_pcts, chunk_counts, finalpcts=finalpcts, pcts=pcts, axis=axis, drop_axis=axis+1, chunks=chunks, meta=numpy.array((), dtype=array.dtype))

    return merged_pcts


def approx_percentile(array, pcts, dim=None, axis=None):
    """
    Get the approximate percentiles of a array along 'axis'

    Args:
        array: Input array (xarray.DataArray, dask.array.Array)
        pcts: List of percentiles to calculate, within the interval [0,100]
        axis: Axis to reduce along

    Returns:
        Array of same type as 'array' with first axis the percentiles from 'pcts',
        remaining axes from 'array' reduced along 'axis'
    """

    if isinstance(array, xarray.DataArray):
        if dim is None and axis is None:
            raise Exception("Please supply one of 'axis' or 'dim'")

        if axis is None:
            axis = array.get_axis_num(dim)

        data = approx_percentile(array.data, pcts, axis=axis)
        dims = ['percentile', *array.dims[:axis], *array.dims[axis+1:]]
        coords = {k: c for k, c in array.coords.items() if array.dims[axis] not in c.dims}
        coords['percentile'] = pcts

        return xarray.DataArray(data, dims=dims, coords=coords, name=array.name)

    if axis is None:
        raise Exception("Please supply 'axis'")

    if isinstance(array, dask.array.Array):
        return dask_approx_percentile(array, pcts, axis=axis)

    return numpy.percentile(array, pcts, axis=axis)
###########################################################
###########################################################


###########################################################
###########################################################
def checkpoint(ctime):
  import time

  """ Computes the spent time from the last checkpoint

  Input: a given time
  Output: present time
  Print: the difference between the given time and present time
  Author: Alejandro Di Luca
  Created: 07/08/2013
  Last Modification: 14/08/2013

  """
  if ctime==0:
    ctime=time.time()
    dtime=0
  else:
    dtime=time.time()-ctime
    ctime=time.time()
    print('======> DONE in ',float('%.2g' %(dtime)),' seconds',"\n")
  return ctime


###############################################################################
##### __main__  scope
###############################################################################

if __name__ == "__main__":

    main()


###############################################################################
