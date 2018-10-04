#!/usr/bin/env bash

python3 make_dataset.py -p 10 -s .2 -d 1 -t erdos --folder test

python3 run_experiments.py -n 100 -b 1 -k 7 --verbose 1 -s .1 -i gauss --folder test --strategy random-smart
