#!/bin/bash
if [ $# -ne 1 ] ; then
        echo "use as $0 inputFilelist"
        exit 1;
fi

for f in `cat $1`; do
        sbatch  -p amd_512 -N 1  -n 1 -c 128 ./job.sh "$f"
done
