#!/bin/bash

# Script to submit several jobs with dependencies at once in a Slurm system.
# When a list of jobs is submitted, a given job does not start until the previous one finishes.  
set -xe

# List and array of jobs
list=$(ls runwrf_201?_??_??.deck)
list_ar=($list)

# First job to submit and job id
job_sub=$(sbatch ${list_ar[0]})
id_sub=$(echo $job_sub | awk -F ' ' '{print $NF}')

# Rest of sequentially dependent jobs
for i_list in ${list_ar[@]:1:38}; do
    job_sub=$(sbatch --dependency=afterany:$id_sub $i_list)
    id_sub=$(echo $job_sub | awk -F' ' '{print $NF}')
done
