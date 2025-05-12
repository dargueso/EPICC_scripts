#!/usr/bin/env python
"""
#####################################################################
# Author: Daniel Argueso <daniel>
# Date:   2021-03-23T12:50:44+01:00
# Email:  d.argueso@uib.es
# Last modified by:   daniel
# Last modified time: 2021-03-23T12:50:47+01:00
#
# @Project@ EPICC
# Version: 1.0
# Description:
#
# Dependencies: It needs list of files with models available at NCI (generated with SearchCMIPData_NCI.py)
#
# Files:
#
#####################################################################
"""
import os
import subprocess as subprocess

variables=['ua','va','zg','hur','ta','uas','vas','tas','ts','hurs','ps','psl']
experiments = ['historical','ssp585']
create_intersection = False #IF files with interesctions were created, set to false

if create_intersection == True:

    dataset_ref= []
    dataset_new = []

    for expn in experiments:
        #LOOK FOR MDOELS AVAILABLE FOR ALL VARIABLES
        for vn,varname in enumerate(variables):
            print(varname)
            filein = open (f'SearchLocation_{varname}_cmip6_{expn}_mon.txt',"r")


            for line in filein.readlines():
                model = line.split('cmip6_')[0].strip().split('_')[0]
                member = line.split('cmip6_')[0].strip().split('_')[1]
                exper = line.split('cmip6_')[0].strip().split('_')[2]
                model_str = line.split('cmip6_')[0].strip()
                path = line.split('cmip6_')[1].strip()

                if vn == 0:
                    dataset_ref.append(model_str)
                else:
                    dataset_new.append(model_str)

            if vn !=0:
                dataset_int=list(set.intersection(set(dataset_ref),set(dataset_new)))
                dataset_ref=dataset_int.copy()
                dataset_new=[]


            with open(f'SearchLocations_intersection_cmip6_{expn}_mon.txt', 'w') as fout:
                for listitem in dataset_ref:
                    fout.write('%s\n' % listitem)

    #LOOK FOR MDOELS AVAILABLE FOR BOTH EXPERIMENTS (PRESENT AND FUTURE)
    file_hist = open('SearchLocations_intersection_cmip6_historical_mon.txt', 'r')
    file_ssp585 = open('SearchLocations_intersection_cmip6_ssp585_mon.txt', 'r')
    data_hist=[]
    data_ssp585=[]

    for line in file_hist.readlines():
        model = line.split('historical')[0]
        data_hist.append(model)

    for line in file_ssp585.readlines():
        model = line.split('ssp585')[0]
        data_ssp585.append(model)

    data_int = list(set.intersection(set(data_hist),set(data_ssp585)))

    with open(f'SearchLocations_intersection_cmip6_mon.txt', 'w') as fout:
        for listitem in data_int:
            fout.write('%s\n' % listitem)

#LOAD DATA AVAILABLE FOR ALL EXPERIMENTS AND VARIABLES
data_int = []
file_models = open(f'SearchLocations_intersection_cmip6_mon.txt', 'r')
for line in file_models.readlines():
    data_int.append(line.strip())

for expn in experiments:
    for vn,varname in enumerate(variables):

        filein = open (f'SearchLocation_{varname}_cmip6_{expn}_mon.txt',"r")
        for line in filein.readlines():
            dataset = line.split(f'{expn}')[0]
            model = line.split('cmip6_')[0].strip().split('_')[0]
            member = line.split('cmip6_')[0].strip().split('_')[1]
            exper = line.split('cmip6_')[0].strip().split('_')[2]

            if dataset in data_int:
                path_remote = line.split('cmip6_')[1].strip()
                if not os.path.exists(f'{exper}/{model}_{member}/'):
                    os.makedirs(f'{exper}/{model}_{member}/')
                #os.system(f'rsync -avz --progress --stats dxa561@gadi.nci.org.au:/{path_remote}/* {exper}/{model}_{member}/')
                print (f'rsync -avz --progress --stats dxa561@gadi.nci.org.au:/{path_remote}/* {exper}/{model}_{member}/')
                subprocess.call(f'rsync -avz --progress --stats dxa561@gadi.nci.org.au:/{path_remote}/* {exper}/{model}_{member}/',shell=True)
