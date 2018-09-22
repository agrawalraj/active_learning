#!/usr/bin/env bash

python3 make_dataset.py -p 12 -s .5 -d 1 -t erdos --folder twenty_sparse
python3 run_experiments.py -n 12 -b 1 -k 2 --folder twenty_sparse --strategy entropy
python3 run_experiments.py -n 12 -b 1 -k 2 --folder twenty_sparse --strategy random
python3 run_experiments.py -n 12 -b 2 -k 2 --folder twenty_sparse --strategy entropy
python3 run_experiments.py -n 12 -b 2 -k 2 --folder twenty_sparse --strategy random

python3 run_experiments.py -n 24 -b 1 -k 2 --folder twenty_sparse --strategy entropy
python3 run_experiments.py -n 24 -b 1 -k 2 --folder twenty_sparse --strategy random
python3 run_experiments.py -n 24 -b 2 -k 2 --folder twenty_sparse --strategy entropy
python3 run_experiments.py -n 24 -b 2 -k 2 --folder twenty_sparse --strategy random

python3 run_experiments.py -n 48 -b 1 -k 2 --folder twenty_sparse --strategy entropy
python3 run_experiments.py -n 48 -b 1 -k 2 --folder twenty_sparse --strategy random
python3 run_experiments.py -n 48 -b 2 -k 2 --folder twenty_sparse --strategy entropy
python3 run_experiments.py -n 48 -b 2 -k 2 --folder twenty_sparse --strategy random

python3 get_parent_probs.py --folder twenty_sparse --target 6
tar -czf data/twenty_sparse.tar.gz data/twenty_sparse

