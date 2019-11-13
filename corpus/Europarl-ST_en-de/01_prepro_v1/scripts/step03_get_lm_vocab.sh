#!/bin/bash

set -e

export LC_ALL=C.UTF-8

LEX=/scratch/translectures/systems/asr/en/TTP-Jun19/models/lm/mono.lex

DIR=data
mkdir -p $DIR

cat $LEX |
    tail -n+2 |
    awk '{print $1}' |
    sort -u > $DIR/lm.vocab
