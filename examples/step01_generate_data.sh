#!/bin/bash

set -e

export LC_ALL=C.UTF-8

MAXLEN=15
WS=4

for SET in train dev test 
do
    echo "Generating $SET ..."
    for i in $SET/*.txt
    do
        ../scripts/get_samples.py $MAXLEN $WS $PWD/lm.vocab < $i
    done > $SET.ML${MAXLEN}.WS${WS}.txt
done

cat train.ML15.WS4.txt | awk '{for (i=1;i<=NF;i++) t[$i]++} END{for (v in t) print v,t[v] }' > train.vocab.txt
