#!/usr/bin/env bash

python3 make_dataset.py -p 25 -d 1 -t chain --folder test

python3 run_experiments.py -n 100 -b 1 -k 5 -m 5 --verbose 1 -s .1 -i gauss --folder test --strategy entropy-dag-collection
