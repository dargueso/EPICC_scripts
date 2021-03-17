#!/usr/bin/env python
"""
#####################################################################
# Author: Daniel Argueso <daniel>
# Date:   2021-02-15T15:32:16+01:00
# Email:  d.argueso@uib.es
# Last modified by:   daniel
# Last modified time: 2021-03-12T23:04:52+01:00
#
# @Project@ EPICC
# Version: 1.0
# Description:Decks to generate scripts and namelist to run consecutive WRF
# experiments in MareNostrum 4
# See full description below.
# Dependencies:
#
# Files:
#
#####################################################################
"""


# MODIFICATION FOR FONER FROM DECKS CREATED FOR MARENOSTRUM4
#-------------------------------------------------------------------------------

# Write out all the daily scripts to run WRF simulations on marenostrum4 (mn1.bsc.es).
# These scripts use boundary conditions files that were already generated
# locally at UIB using megacelula4 and stored at eady3 (Meteo-UIB)
#
#
# Each created script (runwrf_YEAR_MONTH_DAY.deck) will create and launch 3 dependant
#
#
# 1st to grab the boundary condition files from Meteo-UIB machine
# 2nd to create the namelist.input and run WRF
# 3rd to send the output back to Meteo-UIB
#
#
# The namelist part in runwrf_foner_mc.deck must be updated and consistent with the
# namelist used to create the boundary conditions.
#
# To run the generated scripts, put them into the WRF/run directory on foner, and
# type ./runwrf_yyyy_mm_dd.deck
# the first one of them all if its a cold start or another one if its a continuation
#
#
# At the end of the simulation, it should call the next one.
#
#
# List of variables to be set before running the script:
#---------------------------------------------
# start_month/start_year: First month of the simulation. Check the first day it starts.
# end_month/end_year: Last month of the simulation
# ssh user to transfer data from/to Meteo-UIB
# source_bdy: Path to the boundary conditions on eady2 (Meteo-UIB)
# email: your email address
#
#
# To customize the LSF scripts.
#------------------------------
# This python script recognises template texts (i.e. text framed by "%" signs) in the
# runwrf_foner_mc.deck, and replace this text by customize text calculated in this python script.
# It is possible to add lines in the template csh script that use these template texts.
# The template text in these new lines will be automatically replaced. Here is a list of
# the template texts supported:
#
#
# Note: in the following, the date (month/year) for which we create the csh
# script is called "simulated date".
#%source_bdy%: replaced by the value of MeteoUIB_dir
#%expdir%:       replaced by the value of exp_dir
#%globus_user%:       replaced by the value of MeteoUIB_user
#%email%:    replaced by the value of email
#%project%:       replaced by the value of project
#
#%syear%:        replaced by the year of the simulated date
#%smonth%:       replaced by the month of the simulated date
#%sday%:        replaced by the first day of the simulated date
#%nyear%:        replaced by the year of the simulated date + 1 month
#%nmonth%:       replaced by the month of the simulated date + 1 month
#%eday%:        replaced by the last day of the simulation

# List of template texts with very specific meanings that should be treated carefully:
#%sexist%:     start of an optional section
#%eexist%:     end of an optional section
#%spart%:     start of an optional section
#%epart%:     end of an optional section
#-----------------------------------------


import os
import datetime as dt
import calendar

# Start month of the simulation. Will start at day 1.
start_month = 8
start_year = 2020

# End month of the simulation (included).
end_month =9 
end_year = 2020

# If starting from scratch (not a continuation run)
isrestart = False

#How many days in advance (spin-up)?
spinup=10

#Number of consecutive days that the simulation is split into
lendays=10
#name the input deck to use
indeck = "runwrf_marenostrum_EPICC_2km_ERA5_HVC_GWD.deck"

#username on system and address of the machine containing the bdy files
#BDY_user = "dargueso@130.206.30.86"      # rsync
#Path containing the boundary files.
#BDY_dir = "/home/dargueso/BDY_DATA/ERA5/WRF-boundary/REHIPRE/Original_ERA5"
BDY_dir = "/gpfs/projects/uib33/WRF_BDY/EPICC_2km_ERA5_HVC_GWD"

#scp flags required e.g. for port 6512 need "-P 6512"
BDY_scpflags = " "

#username on system and address of the machine with the restart files
#RST_user = "dargueso@130.206.30.86"
#Path containing the restart files.
#RST_dir = "/home/dargueso/WRF_OUT/REHIPRE/Original_ERA5/restart"
RST_dir = "/gpfs/projects/uib33/WRF_OUT/EPICC/EPICC_2km_ERA5_HVC_GWD/restart"
#scp flags required e.g. for port 6512 need "-P 6512"
RST_scpflags = ""

#username on system and address of the machine to put output files
#OUT_user = "dargueso@130.206.30.86"
#Path containing the boundary files.
#OUT_dir = "/home/dargueso/WRF_OUT/REHIPRE/Original_ERA5/out"
OUT_dir = "/gpfs/projects/uib33/WRF_OUT/EPICC/EPICC_2km_ERA5_HVC_GWD/out"
#scp flags required e.g. for port 6512 need "-P 6512"
OUT_scpflags = " "


#email: to receive an email at the end of each script
email = "d.argueso@uib.es"

#************* end of user input

year = start_year
month = start_month

init_date = dt.datetime(start_year,start_month,1)-dt.timedelta(days=spinup)

year = init_date.year
month= init_date.month
day = init_date.day

while (year < end_year or (year == end_year and month < end_month)):

    #get the month as a 2 digit string
    monthstr = str(month).rjust(2,"0")

    #Number of days in this month
    numdays = calendar.monthrange(year,month)[1]


    s_simyea = year
    s_simmon = month
    s_simday = day

    end_date = dt.datetime(year,month,day) + dt.timedelta(days=lendays)
    if end_date>dt.datetime(end_year,end_month,1):
        end_date = dt.datetime(end_year,end_month,1)

    e_simyea = end_date.year
    e_simmon = end_date.month
    e_simday = end_date.day

    sdaystr = str(s_simday).rjust(2,"0")
    smonstr = str(s_simmon).rjust(2,"0")
    edaystr = str(e_simday).rjust(2,"0")
    emonstr = str(e_simmon).rjust(2,"0")

    #open the sample deck
    fin = open (indeck,"r")

    #open the deck I am creating
    fout = open ("runwrf_%s_%s_%s.sh"%(year,smonstr,sdaystr),"w")

    # make sure the output file is executable by user
	#os.fchmod(fout.fileno(),0744)
    #Loop over the lines of the input file
    for lines in fin.readlines():

        # Replace template fields by values
        lines = lines.replace("%BDYdir%", BDY_dir)
        #lines = lines.replace("%BDYuser%", BDY_user)
        lines = lines.replace("%OUTdir%", OUT_dir)
        #lines = lines.replace("%OUTuser%", OUT_user)
        lines = lines.replace("%RSTdir%", RST_dir)
        #lines = lines.replace("%RSTuser%", RST_user)
        lines = lines.replace("%email%", email)
        lines = lines.replace("%syear%", str(s_simyea))
        lines = lines.replace("%smonth%", smonstr)
        lines = lines.replace("%sday%", sdaystr)
        lines = lines.replace("%eyear%", str(e_simyea))
        lines = lines.replace("%emonth%", emonstr)
        lines = lines.replace("%eday%", edaystr)
        lines = lines.replace("%isrestart%", "."+str(isrestart).lower()+".")
        lines = lines.replace("%resint%", str(lendays*1440))

        fout.write(lines)
    #Close input and output files
    fin.close()
    fout.close()
    # All other runs are restarts
    isrestart=True

    year  = end_date.year
    month = end_date.month
    day   = end_date.day
