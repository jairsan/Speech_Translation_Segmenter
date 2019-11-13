#!/bin/bash

set -e

export LC_ALL=C.UTF-8

SPLITTER=/scratch/jiranzotmp/trabajo/Europarl-ST/03_align/sentence_splitter_tool/txt2txt-sentences.py
PREPRO=~/wds/asr-scripts/lm/en/2018/prepro.sh

for SET in train dev test
do
    DIR=norm/$SET
    mkdir -p $DIR
    for i in originals/$SET/*.txt
    do
        echo "Processing $i ..."
        TO=$DIR/`basename $i`
        python3 $SPLITTER $i en |
            sed "s| ’ \(s\)|'\1|g" |
            $PREPRO > $TO
    done
done
