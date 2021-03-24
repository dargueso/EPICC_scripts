#!/usr/bin/env python
"""
#####################################################################
# Author: Daniel Argueso <daniel> @ UIB
# Date:   2018-02-13T09:44:23+11:00
# Email:  d.argueso@uib.es
# Last modified by:   daniel
# Last modified time: 2018-02-13T09:44:26+11:00
#
# @Project@ atmopy
# Version: x.0 (Beta)
# Description:
# Small modules containing tools to plot and process data
# Derived from soest_utils and uib_utils.py
#
# Dependencies:numpy,re,cPickle
#
# Files:
#
#####################################################################
"""
import numpy as np
from mylibs.constants import const as const

###########################################################
###########################################################

# Write out and read using cPickle to make testing efficiently

def write_pkl(var,filename):
    import pickle as pkl
    handle=open('%s' % filename,'wb')
    pkl.dump(var,handle)
    handle.close

def read_pkl(filename):
    import pickle as pkl
    handle=open('%s' % filename,'rb')
    var=pkl.load(handle)
    return var



###########################################################
###########################################################
#### PLOTTING ####
###########################################################
###########################################################

def rgb2cmap(filename,base='255'):
    """Function to read a rgb file (i.e. NCL colortables) and convert it
     to matplotlib colormap
     Author: Daniel Argueso @ CCRC, UNSW. Sydney (Australia)
    """

    from matplotlib.colors import ListedColormap
    import re

    filein=open(filename)
    lines=filein.readlines()
    colors=[]

    for line in lines[2:]:
        line=re.sub('\s+',' ',line)
        li=line.strip()
        if li:
            values=li.split(' ')
            if base == '255':
              new_values=[i/255. for i in map(int,values[:3])]
            else:
              new_values=[i for i in map(float,values[:3])]
            colors.append(new_values)
    cmap=ListedColormap(colors)
    cmap.set_over(colors[-1])
    cmap.set_under(colors[0])

    return cmap

###########################################################
###########################################################

def setnicescale(minval,maxval,numdivs=15,symmetry=False):
    """Function to generate a nice scale given a minimum and maximum value
        Similarly to NCL it uses a set of relatively round numbers 1.0,2.0,2.5,4.0,5.0
        and adjust the scale using this set and powers of 10.
        minval: minimum value that will be plotted
        maxval: maximum value that will be plotted
        numdiv: number of divisions in the scale (to set the stride)
        symmetry: regardles of the min and max, whether the scale must be symmetrical with respect to 0
        ---
        minlim: lower limit of the scale
        maxlim: upper limit of the scale
        stride: stride of the scale

        Author: Daniel Arguerso
    """
    import numpy as np

    setnum=np.asarray([1.0,1.5,2.0,2.5,3.0,4.0,5.0,6.0,8.0,10.0],dtype=np.float)
    setnum_forstrides=np.asarray([1.0,2.0,2.5,4.0,5.0,10.0],dtype=np.float)
    if maxval==0:
      maxval=0.
    elif maxval<0:
      val=-maxval
      dist_setmaxval=setnum*(10**np.floor(np.log10(val)))-val
      maxlim=-setnum[list(dist_setmaxval).index(np.max(dist_setmaxval[dist_setmaxval<0]))]*(10**np.floor(np.log10(val)))
    else:
      val=maxval
      #print val
      dist_setmaxval=setnum*(10**np.floor(np.log10(val)))-val
      #print setnum*(10**np.floor(np.log10(val)))
      maxlim=setnum[list(dist_setmaxval).index(np.min(dist_setmaxval[dist_setmaxval>=0]))]*(10**np.floor(np.log10(val)))

    if minval==0:
      minlim=0.
    elif minval<0:
      minval=-minval
      dist_setminval=setnum*(10**np.floor(np.log10(minval)))-minval
      minlim=-setnum[list(dist_setminval).index(np.min(dist_setminval[dist_setminval>0]))]*(10**np.floor(np.log10(minval)))
    else:
      dist_setminval=minval-setnum*(10**np.floor(np.log10(minval)))
      minlim=setnum[list(dist_setminval).index(np.min(dist_setminval[dist_setminval>=0]))]*(10**np.floor(np.log10(minval)))

    if np.sign(minlim)==np.sign(maxlim):
      if (minlim>0.) & (maxlim/minlim>2.5):
        minlim=0.
      if (minlim<0.) & (minlim/maxlim>2.5):
        maxlim=0.


    if symmetry==True:
      print ("symmetrical scale")
      bothlim=max(np.abs(minlim),np.abs(maxlim))
      maxlim=bothlim
      minlim=-bothlim
    stride_scale=setnum_forstrides*(10**(np.floor(np.log10(np.abs(maxlim-minlim)))-1))
    nicediv=np.abs(maxlim-minlim)/numdivs

    ndiv=numdivs
    #print ndiv
    #print stride_scale, nicediv
    #print minval,maxval
    stride=nicediv
    while ndiv>5:
      if (stride_scale % nicediv==0).any():
        stride=stride_scale[np.where(stride_scale % nicediv==0.)[0][0]]
        #print "%s divisions is more appropriate" %(ndiv)
        break
      ndiv=ndiv-1
      nicediv=np.abs(maxlim-minlim)/ndiv

    if symmetry==True:
      maxlim=maxlim+stride

    mylevels=np.arange(minlim,maxlim,stride)


    #print minlim,maxlim,stride
    return mylevels


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


###########################################################
###########################################################
#########  SMALL UTILS to PROCESS DATA ####################
###########################################################
###########################################################


def calc_distance_coord(lat1,lon1,lat2,lon2):
    """Function to calculate distance using coordinates
       This uses the spherical law of cosines
       lat1,lat2,lon1,lon2 : input coordinates (degrees)
       d : distance (m)"""
    lat1r = lat1*np.pi/180.
    lat2r = lat2*np.pi/180.
    lon1r = lon1*np.pi/180.
    lon2r = lon2*np.pi/180.
    d = np.arccos(np.sin(lat1r)*np.sin(lat2r) + np.cos(lat1r)*np.cos(lat2r)*np.cos(lon2r-lon1r))*const.earth_radius

    return d


def eq_recta(x1,y1,x2,y2):

    a=(y2-y1)/(x2-x1)

    b = y1-a*x1

    return a,b

def shift_LST_2D(var,LST_shift):

    """Function to displace hours from UTC (WRF output) to Local Solar time
       using a 2D array with LST shifts (time, lon).
       The variable to shift is expected to have (time,lat,lon) dimensions.
    """

    this_shift=LST_shift[0]
    var = np.roll(var,this_shift,axis=0)

    n_1=0
    while n_1<len(LST_shift):
        this_shift+=1
        n = np.argwhere(LST_shift==this_shift)[0][0]
        n_1 = np.argwhere(LST_shift==this_shift)[-1][0]+1
        var[:,:,n:]=np.roll(var[:,:,n:],1,axis=0)

    return var

def shift_LST_3D(var,LST_shift):

    """Function to displace hours from UTC (WRF output) to Local Solar time
       using a 3D array with LST shifts (time, lon).
       The variable to shift is expected to have (time,level,lat,lon) dimensions.
    """

    this_shift=LST_shift[0]
    var = np.roll(var,this_shift,axis=0)

    n_1=0
    while n_1<len(LST_shift):
        this_shift+=1
        n = np.argwhere(LST_shift==this_shift)[0][0]
        n_1 = np.argwhere(LST_shift==this_shift)[-1][0]+1
        var[:,:,:,n:]=np.roll(var[:,:,:,n:],1,axis=0)

    return var
