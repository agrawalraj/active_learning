#!/usr/bin/env bash

python3 make_dataset.py -p 51 -d 25 -t chain --folder chain_test8

python3 run_experiments.py -n 180 -b 1 -k 1 -s 1 -i gauss --folder chain_test8 --strategy entropy-dag-collection
python3 run_experiments.py -n 180 -b 2 -k 1 -s 1 -i gauss --folder chain_test8 --strategy entropy-dag-collection
python3 run_experiments.py -n 180 -b 3 -k 1 -s 1 -i gauss --folder chain_test8 --strategy entropy-dag-collection

python3 run_experiments.py -n 180 -b 1 -k 1 -s 1 -i gauss --folder chain_test8 --strategy entropy-dag-collection
python3 run_experiments.py -n 180 -b 2 -k 2 -s 1 -i gauss --folder chain_test8 --strategy entropy-dag-collection
python3 run_experiments.py -n 180 -b 3 -k 3 -s 1 -i gauss --folder chain_test8 --strategy entropy-dag-collection

python3 run_experiments.py -n 180 -b 1 -k 1 -s 1 -i gauss --folder chain_test8 --strategy entropy-dag-collection
python3 run_experiments.py -n 180 -b 2 -k 2 -s 1 -i gauss --folder chain_test8 --strategy entropy-dag-collection
python3 run_experiments.py -n 180 -b 3 -k 3 -s 1 -i gauss --folder chain_test8 --strategy entropy-dag-collection


