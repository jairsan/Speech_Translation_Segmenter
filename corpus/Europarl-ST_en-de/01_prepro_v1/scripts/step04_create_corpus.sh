#!/bin/bash

set -e

export LC_ALL=C.UTF-8

MAXLEN=15
WS=4

DIR=corpus
mkdir -p $DIR

for SET in dev test train
do
    echo "Generating $SET ..."
    for i in norm/$SET/*.txt
    do
        scripts/get_samples.py $MAXLEN $WS data/lm.vocab < $i
    done > $DIR/$SET.ML${MAXLEN}.WS${WS}.txt
done
