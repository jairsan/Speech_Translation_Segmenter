#!/bin/bash

set -e

export LC_ALL=C.UTF-8

ODATA=/scratch/translectures/data/Europarl-ST/RELEASES/v1.0/en/de

for SET in train dev test
do
    DIR=originals/$SET
    mkdir -p $DIR
    N=1
    cat $ODATA/$SET/speeches.en |
        while read l
        do
            NAME=`printf "%05d" $N`
            F=$DIR/en-de.$NAME.txt
            echo "$l" > $F
            N=$[N+1]
        done
done
