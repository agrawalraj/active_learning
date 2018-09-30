#!/usr/bin/env bash

python3 make_dataset.py -p 20 -d 50 -t chain --folder chain_test
python3 run_experiments.py -n 256 -b 1 -k 1 -s 2 --folder chain_test --strategy entropy-dag-collection
python3 run_experiments.py -n 256 -b 2 -k 1 -s 2 --folder chain_test --strategy entropy-dag-collection
python3 run_experiments.py -n 256 -b 1 -k 2 -s 2 --folder chain_test --strategy entropy-dag-collection
python3 run_experiments.py -n 256 -b 2 -k 2 -s 2 --folder chain_test --strategy entropy-dag-collection
python3 run_experiments.py -n 1024 -b 1 -k 1 -s 2 --folder chain_test --strategy entropy-dag-collection
python3 run_experiments.py -n 1024 -b 2 -k 1 -s 2 --folder chain_test --strategy entropy-dag-collection
python3 run_experiments.py -n 1024 -b 1 -k 2 -s 2 --folder chain_test --strategy entropy-dag-collection
python3 run_experiments.py -n 1024 -b 2 -k 2 -s 2 --folder chain_test --strategy entropy-dag-collection

python3 run_experiments.py -n 256 -b 1 -k 1 -s 2 --folder chain_test --strategy random
python3 run_experiments.py -n 256 -b 2 -k 1 -s 2 --folder chain_test --strategy random
python3 run_experiments.py -n 256 -b 1 -k 2 -s 2 --folder chain_test --strategy random
python3 run_experiments.py -n 256 -b 2 -k 2 -s 2 --folder chain_test --strategy random
python3 run_experiments.py -n 1024 -b 1 -k 1 -s 2 --folder chain_test --strategy random
python3 run_experiments.py -n 1024 -b 2 -k 1 -s 2 --folder chain_test --strategy random
python3 run_experiments.py -n 1024 -b 1 -k 2 -s 2 --folder chain_test --strategy random
python3 run_experiments.py -n 1024 -b 2 -k 2 -s 2 --folder chain_test --strategy random


