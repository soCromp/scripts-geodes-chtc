#!/bin/bash

TZ="America/Chicago"
export PATH=/usr/sbin:$PATH
echo 'Date: ' `date`
echo 'Host: ' `hostname`
echo 'System: ' `uname -spo`

# arguments passed from sub file
RUNNAME=$1
PROCESS=$2
inname=$3
outname=$4

python swa_merge.py $inname --out $outname

