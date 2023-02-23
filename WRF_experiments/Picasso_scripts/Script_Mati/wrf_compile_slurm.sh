#!/bin/sh
#
# <wrf_compile_wrf.sh>
#
# Job to compile WRF in a SLURM environment.
#

#------------------------ JOB PREAMBLE ------------------------#

# The name to show in queue lists for this job:
#SBATCH -J compilation_intel

# Number of desired cores:
#SBATCH --nodes=1
#
#SBATCH --ntasks-per-node=1

# Amount of RAM needed for this job:
#SBATCH --mem=100gb

# The time the job will be running, 10 hours:
#SBATCH --time=5:00:00

# To use GPUs you have to request them:
##SBATCH --gres=gpu:1

# If you need nodes with special features uncomment the desired constraint line:
##SBATCH --constraint=bigmem
##SBATCH --constraint=cal

# Set output and error files
#SBATCH --error=compile_WRF.err
#SBATCH --output=compile_WRF.out

# MAKE AN ARRAY JOB, SLURM_ARRAYID will take values from 1 to 100
# Commented:
##SARRAY --range=1-100


#------------------------ JOB BODY ------------------------#

module load intel/2021.4

source $SCRATCH/WRFEnv/scripts/envars_setup.sh
cd $WRF_DIR
ulimit -s unlimited
./compile em_real 2>&1 | tee compile.log

