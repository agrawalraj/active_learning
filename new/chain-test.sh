#!/usr/bin/env bash

python3 make_dataset.py -p 20 -d 1 -t chain --folder chain_test6

python3 run_experiments.py -n 2048 -b 1 -k 1 -s 2 --folder chain_test6 --strategy entropy-dag-collection
python3 run_experiments.py -n 2048 -b 1 -k 2 -s 2 --folder chain_test6 --strategy entropy-dag-collection
python3 run_experiments.py -n 2048 -b 2 -k 1 -s 2 --folder chain_test6 --strategy entropy-dag-collection
python3 run_experiments.py -n 2048 -b 2 -k 2 -s 2 --folder chain_test6 --strategy entropy-dag-collection
python3 run_experiments.py -n 2048 -b 3 -k 1 -s 2 --folder chain_test6 --strategy entropy-dag-collection
python3 run_experiments.py -n 2048 -b 3 -k 2 -s 2 --folder chain_test6 --strategy entropy-dag-collection

python3 run_experiments.py -n 2048 -b 1 -k 1 -s 2 --folder chain_test6 --strategy random
python3 run_experiments.py -n 2048 -b 1 -k 2 -s 2 --folder chain_test6 --strategy random
python3 run_experiments.py -n 2048 -b 2 -k 1 -s 2 --folder chain_test6 --strategy random
python3 run_experiments.py -n 2048 -b 2 -k 2 -s 2 --folder chain_test6 --strategy random
python3 run_experiments.py -n 2048 -b 3 -k 1 -s 2 --folder chain_test6 --strategy random
python3 run_experiments.py -n 2048 -b 3 -k 2 -s 2 --folder chain_test6 --strategy random

