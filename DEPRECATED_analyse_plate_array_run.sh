#!/bin/bash

#this file is an workaround to get the environment shell variable LSB_JOBINDEX and pass it to the python script

#index=$LSB_JOBINDEX
python3 cluster.py $* -x $LSB_JOBINDEX

# indir=$1 
# outdir=$3
# loop=$2
# channel=$3
# wells=$4
# crop=$4
# average=$5
# threads=$6

# python3 cluster.py -i $indir -l $loop -o $outdir $crop -ix $index -a $average -p $threads