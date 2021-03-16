#!/usr/bin/env python

"""
#####################################################################
# Author: Daniel Argueso <daniel> @ UIB
# Date:   2018-02-14T13:30:48+11:00
# Email:  d.argueso@uib.es
# Last modified by:   daniel
# Last modified time: 2021-03-16T17:48:56+01:00
#
# @Project@ EPICC_2km_ERA5_HVC_GWD
# Version: 1.0
# Description: Script to generate postprocessed files from EPICC wrfouts and wrfprecs
#
# Dependencies: atmopy
#
# Files: wrfout_* or wrfprec_*from EPICC runs
#
#####################################################################
"""

import netCDF4 as nc
import numpy as np
from glob import glob
import datetime as dt
import calendar
import os
import atmopy as ap
from optparse import OptionParser
import re
from dateutil.relativedelta import relativedelta
import sys

from joblib import Parallel, delayed

import atmopy.compute_vars as cvars
from atmopy.wrf_utils import wrftime2date,sel_wrfout_files



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

def read_input(filename):
    """Function to read namelist and pass the arguments on to the postprocess
    """

    filein = open(filename,'r')
    options,sentinel,varnames=filein.read().partition('#### Requested output variables (DO NOT CHANGE THIS LINE) ####')
    fileopts=open('fileopts.tmp','w')
    fileopts.write(options)
    filevarnames=open('filevarnames.tmp','w')
    filevarnames.write(varnames)

    fileopts=open('fileopts.tmp','r')
    lines=fileopts.readlines()

    inputinf={}
    entryname=[]
    entryvalue=[]
    for line in lines:
      line=re.sub('\s+',' ',line)
      line=line.replace(" ", "")
      li=line.strip()
      #Ignore empty lines
      if li:
        #Ignore commented lines
        if not li.startswith("#"):
          values=li.split('=')
          entryname.append(values[0])
          entryvalue.append(values[1])
    fileopts.close()

    for ii in range(len(entryname)):
      inputinf[entryname[ii]]=entryvalue[ii]

    filevarnames=open('filevarnames.tmp','r')
    lines=filevarnames.readlines()
    varnames=[]
    for line in lines:
      line=re.sub('\s+',' ',line)
      line=line.replace(" ", "")
      li=line.strip()
      #Ignore empty lines
      if li:
        #Ignore commented lines
        if not li.startswith("#"):
          values=li.split()
          varnames.append(values[0])
    filevarnames.close()

    os.remove('filevarnames.tmp')
    os.remove('fileopts.tmp')

    print('Variables that will be obtained from postprocessing:',varnames)
    return inputinf,varnames

###########################################################
###########################################################

def postproc_var(inputinf,varn):

    wrun = inputinf['wrun']
    institution=inputinf['institution']
    path_in = inputinf['path_in']
    path_out = inputinf['path_out']
    patt    = inputinf['patt']
    syear = int(inputinf['syear'])
    eyear = int(inputinf['eyear'])
    smonth = int(inputinf['smonth'])
    emonth = int(inputinf['emonth'])
    dom = inputinf['dom']

    ctime_var=checkpoint(0)

    y = syear
    m = smonth

    if not glob('%s/%s_%s*' %(fullpathin,patt,dom)):
        raise Exception("ERROR: no available files in requested directory: %s/%s_%s*" %(fullpathin,patt,dom))

    while (y < eyear or (y == eyear and m <= emonth)):
        for d in range(1,calendar.monthrange(y, m)[1]+1):
            print(y, m, d)

            if not glob('%s/%s_%s_%s*' %(fullpathin,patt,dom,y)):
                raise Exception("ERROR: no available files for year %s requested directory: %s/%s_%s_%s-%s*" %(y,path_in,patt,dom,y,m))

            sdate="%s-%s-%s" %(y,str(m).rjust(2,"0"),str(d).rjust(2,"0"))
            filesin = sorted(glob('%s/%s_%s_%s*' %(fullpathin,patt,dom,sdate)))

            if not filesin:
                print("No available files for month %s" %(sdate))
                continue

            x=[]
            t=[]

            if len(filesin) == 1:
                print(filesin)
                varout,atts = cvars.compute_WRFvar(filesin[0],varn,inputinf)
                otimes =  wrftime2date(filesin[0].split())[:]
            else:

                for n,filename in enumerate(filesin):
                    print(filename)

                    tFragment = wrftime2date(filename.split())[:]
                    xFragment,atts = cvars.compute_WRFvar(filename,varn,inputinf)

                    if len(tFragment)==1:
                        if len(xFragment.shape) == 3:
                            xFragment=np.expand_dims(xFragment,axis=0)
                        if len(xFragment.shape) == 2:
                            xFragment=np.expand_dims(xFragment,axis=0)


                    x.append(xFragment)
                    t.append(tFragment)

                varout = np.concatenate(x, axis=0)
                otimes = np.concatenate(t, axis=0)

            ###########################################################
            ###########################################################

            # ## Creating netcdf files
            fileout = "%s/%s_%s_%s.nc" %(fullpathout,institution,varn,str(sdate))

            ref_file = nc.Dataset(filesin[0])
            lat=ref_file.variables['XLAT'][0,:]
            lon=ref_file.variables['XLONG'][0,:]

            varinfo = { 'values': varout,
                        'varname': varn,
                        'atts':atts,
                        'lat': lat,
                        'lon': lon,
                        'times': otimes}

            cvars.create_netcdf(varinfo,fileout)



            #edate = dt.datetime(y,m,d) + dt.timedelta(days=1)
            print(otimes[-1].strftime("%Y-%m-%d"))
            edate = otimes[-1] + dt.timedelta(days=1)

            y = edate.year
            m = edate.month
            #d = edate.day
    ctime=checkpoint(ctime_var)

###########################################################
###########################################################

def postproc_var_byday(inputinf,varn,date):

    wrun = inputinf['wrun']
    institution=inputinf['institution']
    path_in = inputinf['path_in']
    path_out = inputinf['path_out']
    patt    = inputinf['patt']
    dom = inputinf['dom']
    fullpathin = inputinf['fullpathin']

    ctime_var=checkpoint(0)


    y = date.year
    m = date.month
    d = date.day

    print(y,m,d)

    # if not glob('%s/%s_%s_%s*' %(fullpathin,patt,dom,y)):
    #     raise Exception("ERROR: no available files for year %s requested directory: %s/%s_%s_%s-%s*" %(y,path_in,patt,dom,y,m))

    sdate="%s-%s-%s" %(y,str(m).rjust(2,"0"),str(d).rjust(2,"0"))
    filesin = sorted(glob('%s/%s_%s_%s*' %(fullpathin,patt,dom,sdate)))

    # if not filesin:
    #     print "No available files for day %s" %(sdate)
    #     continue

    x=[]
    t=[]

    if len(filesin) == 1:
        print(filesin)
        varout,atts = cvars.compute_WRFvar(filesin[0],varn,inputinf)
        otimes =  wrftime2date(filesin[0].split())[:]
    else:

        for n,filename in enumerate(filesin):
            #print(filename)

            tFragment = wrftime2date(filename.split())[:]
            xFragment,atts = cvars.compute_WRFvar(filename,varn,inputinf)

            if len(tFragment)==1:
                if len(xFragment.shape) == 3:
                    xFragment=np.expand_dims(xFragment,axis=0)
                if len(xFragment.shape) == 2:
                    xFragment=np.expand_dims(xFragment,axis=0)


            x.append(xFragment)
            t.append(tFragment)

        varout = np.concatenate(x, axis=0)
        otimes = np.concatenate(t, axis=0)

    ###########################################################
    ###########################################################

    # ## Creating netcdf files
    fileout = "%s/%s_%s_%s.nc" %(fullpathout,institution,varn,str(sdate))

    ref_file = nc.Dataset(filesin[0])
    lat=ref_file.variables['XLAT'][0,:]
    lon=ref_file.variables['XLONG'][0,:]

    varinfo = { 'values': varout,
                'varname': varn,
                'atts':atts,
                'lat': lat,
                'lon': lon,
                'times': otimes}

    cvars.create_netcdf(varinfo,fileout)



    #edate = dt.datetime(y,m,d) + dt.timedelta(days=1)
    print(otimes[-1].strftime("%Y-%m-%d"))
    ctime=checkpoint(ctime_var)

###########################################################
###########################################################

# Check initial time
ctime_i=checkpoint(0)
ctime=checkpoint(0)

#### READING INPUT FILE ######
### Options

parser = OptionParser()

parser.add_option("-i", "--infile", dest="infile",
help="file with the input arguments", metavar="INPUTFILE")
(opts, args) = parser.parse_args()

###


#### Reading input info file ######
inputinf,varnames=read_input(opts.infile)


###########################################################
###########################################################



wrun_all = [x.strip() for x in inputinf['wrun'].split(',')]
path_in = inputinf['path_in']
path_out = inputinf['path_out']
syear_all = [x.strip() for x in inputinf['syear'].split(',')]
eyear_all = [x.strip() for x in inputinf['eyear'].split(',')]
smonth = int(inputinf['smonth'])
emonth = int(inputinf['emonth'])


for wrun in wrun_all:
    for n,syear in enumerate(syear_all):
        eyear = eyear_all[n]
        fullpathin = path_in + "/" + wrun + "/out"
        fullpathout = path_out + "/" + wrun + "/" + syear + "-" + eyear
        inputinf['fullpathin']=fullpathin
        inputinf['fullpathout']=fullpathout

        if not os.path.exists(fullpathout):
            os.makedirs(fullpathout)
        datenow=dt.datetime.now().strftime("%Y-%m-%d_%H:%M")
        #logfile = '%s/postprocess_%s_%s_%s-%s_%s_%s.log' %(fullpathout,wrun,syear,smonth,eyear,emonth,datenow)
        #print ('The output messages are written to %s' %(logfile))
        #sys.stdout = open('%s' %(logfile), "w")
        ###########################################################
        ###########################################################



        #Parallel(n_jobs=8)(delayed(postproc_var)(inputinf,varn) for varn in varnames)

        for varn in varnames:

            d1 = dt.datetime(int(syear),smonth,1)
            d2 = dt.datetime(int(eyear),emonth,calendar.monthrange(int(eyear), emonth)[1])+dt.timedelta(days=1)
            total_hours = (d2-d1).days*24+(d2-d1).seconds//3600
            total_days = (d2-d1).days
            date_list= [d1 + dt.timedelta(days=x) for x in range(0, total_days)]
            Parallel(n_jobs=10)(delayed(postproc_var_byday)(inputinf,varn,date) for date in date_list)

        ctime=checkpoint(ctime_i)

        #sys.stdout.close()
