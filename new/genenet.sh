#!/usr/bin/env bash

for k in 1 3 10
do
    for t in 1 6 13 23 32 33 47
    do

    python run_genenet.py -t $t -k $k

    done
done
