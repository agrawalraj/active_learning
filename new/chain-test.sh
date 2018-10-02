#!/usr/bin/env bash

python3 make_dataset.py -p 7 -d 1 -t chain --folder chain_test7

python3 run_experiments.py -n 10 -b 1 -k 1 -s 1 --constant-intervention 1 --folder chain_test7 --strategy entropy-dag-collection

