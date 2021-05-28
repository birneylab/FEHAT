#!/bin/bash

#this file is an workaround to get the environment shell variable LSB_JOBINDEX and pass it to the python script
indir=$1 
loops=$2  
out_dir=$3
crop=$4
average=$5
process=$6

index=$LSB_JOBINDEX

python3 segment_heart.py -i $indir -l $loops -o $out_dir $crop -ix $index -a $average -p $process
