#!/usr/bin/env bash

python3 make_dataset.py -p 10 -d 1 -t chain --folder chain_test
python3 run_experiments.py -n 2048 -b 1 -k 1 -s 2 --folder chain_test --strategy entropy-dag-collection
python3 run_experiments.py -n 2048 -b 1 -k 1 -s 2 --folder chain_test --strategy random


