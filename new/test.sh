#!/usr/bin/env bash

python3 make_dataset.py -p 3 -d 1 -t chain --folder test

python3 run_experiments.py -n 180 -b 1 -k 1 -s 5 -i node-variance --folder test --strategy entropy-dag-collection
