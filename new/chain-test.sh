#!/usr/bin/env bash

python3 make_dataset.py -p 20 -d 50 -t chain --folder chain_test7

python3 run_experiments.py -n 2048 -b 1 -k 1 -s 5 --folder chain_test7 --strategy entropy-dag-collection
python3 run_experiments.py -n 2048 -b 1 -k 2 -s 5 --folder chain_test7 --strategy entropy-dag-collection
python3 run_experiments.py -n 2048 -b 2 -k 1 -s 5 --folder chain_test7 --strategy entropy-dag-collection
python3 run_experiments.py -n 2048 -b 2 -k 2 -s 5 --folder chain_test7 --strategy entropy-dag-collection

python3 run_experiments.py -n 2048 -b 1 -k 1 -s 5 --folder chain_test7 --strategy random
python3 run_experiments.py -n 2048 -b 1 -k 2 -s 5 --folder chain_test7 --strategy random
python3 run_experiments.py -n 2048 -b 2 -k 1 -s 5 --folder chain_test7 --strategy random
python3 run_experiments.py -n 2048 -b 2 -k 2 -s 5 --folder chain_test7 --strategy random

