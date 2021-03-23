#!/usr/bin/env python

""" Script to search fo CMIP5 or CMIP6 models in NCI

Author: Alejandro Di Luca @ CCRC, UNSW. Sydney (Australia)
email: a.diluca@unsw.edu.au
Created: May 2019

IMPORTANT
IMPORTANT
USE THE FOLLOWING MODULES

module use /g/data3/hh5/public/modules
module load conda/analysis3-unstable

"""
import numpy as np
import glob as glob
import sys
import pdb
import copy as cp
import os as os
from clef import code
#from ARCCSSive import CMIP5

# VARIABLES
variables=['ua','va','zg','hur','ta','uas','vas','tas','ts','hurs','ps','psl']
ccrcs='z3444417@maelstrom.ccrc.unsw.edu.au:/srv/ccrc/data30/z3444417/CMIP5/tmax_decomposition/'


# EXPERIMENTS
ensembles=['cmip6']
path_out='/g/data3/w28/dxa561/CMIP6/'

# FREQUENCY
frequency='mon'

db=code.connect()
s = code.Session()
#cmip5 = CMIP5.connect()
for ensemble in ensembles:
  print('ensemble: ', ensemble)

  if ensemble=='cmip5':
    members=['r1i1p1','r2i1p1','r3i1p1','r4i1p1','r5i1p1','r6i1p1','r7i1p1','r8i1p1','r9i1p1','r10i1p1']
    members=['r1i1p1']
    experiments = ["historical","rcp85"]

  if ensemble=='cmip6':
    members=['r1i1p1f1','r1i1p1f2','r1i1p1f3']
#    members=['r1i1p1f1','r1i1p1f2']
    experiments = ["historical","ssp585"]

  for variable in variables:
    print('variable: '+variable)
    if variable in ['orog','sftlf']:
      experiments = ["historical","piControl","abrupt-4xCO2","1pctCO2"]

    for exp in experiments:
      print('experiment: ', exp)
      txt_name='%sSearchLocation_%s_%s_%s_%s%s'%(path_out,variable,ensemble,exp,frequency,'.txt')
      filetxt = open(txt_name,"w")
      for member in members:
        if variable in ['orog','sftlf']:
          frequency='fx'
          if ensemble=='cmip5':
            member='r0i0p0'

        print('member: ', member)
        outputs=code.search(s,project=ensemble,experiment=exp,frequency=frequency,variable=variable,ensemble=member)

        last_model=''
        list_all=[]
        list_w=[]
        for mm in range(len(outputs['source_id'])):

          if ensemble=='cmip5':
            model = m['model']
            files = m['filenames']
            path  = m['pdir']
          if ensemble=='cmip6':
            model = outputs['source_id'][mm]
            files = outputs['filename'][mm]
            path  = outputs['path'][mm]

          print(model)
          list_all.append(model+'_'+member+'_'+exp+'_'+ensemble+'_'+path+'/'+' \n')
          if last_model==model:
            #print(list_all)
            v0=list_all[-2].split('/')[-3]
            v1=list_all[-1].split('/')[-3]
            if v0==np.sort([v0,v1])[-1]:
              list_all.remove(list_all[-1])
            else:
              list_all.remove(list_all[-2])
          else:
            last_model=model

          if variable in ['orog','sftlf']:
            filenames=np.asarray(sorted(glob.glob(path+'/*')))
            if len(filenames)>0:

              filename =filenames[0].split('/')[-1]
              model = filename.split("_")[2]
              run = filename.split("_")[4][:-3]
              freq  = filename.split("_")[1]
              exp = filename.split("_")[3]
              print(path+'/'+filename)
              os.system('rsync -avzL '+path+'/'+filename+' '+ccrcs+variable+'/')

        for mli in list_all:
          print('reading model: ', model)
          print(path)
          filetxt.write(mli)

      print('  --> output file: '+txt_name)
      filetxt.close()
