#!/bin/bash
if [ $# -ne 1 ] ; then
        echo "use as $0 inputfile"
        exit 1;
fi
module load anaconda/3-Python3.7.4-fenggl
source activate py3.7
python -u /public3/home/scg5914/software-scg5914/LSW/1/data/data2/def2svp_b3lyp.py  $1
