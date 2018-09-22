#!/usr/bin/env bash

python3 make_dataset.py -p 20 -s .5 -d 50 -t erdos --folder twenty_sparse
python3 run_experiments.py -n 16 -b 1 -k 2 --folder twenty_sparse --strategy entropy
python3 run_experiments.py -n 16 -b 1 -k 2 --folder twenty_sparse --strategy random
python3 run_experiments.py -n 16 -b 2 -k 2 --folder twenty_sparse --strategy entropy
python3 run_experiments.py -n 16 -b 2 -k 2 --folder twenty_sparse --strategy random
python3 run_experiments.py -n 16 -b 4 -k 2 --folder twenty_sparse --strategy entropy
python3 run_experiments.py -n 16 -b 4 -k 2 --folder twenty_sparse --strategy random

python3 run_experiments.py -n 64 -b 1 -k 2 --folder twenty_sparse --strategy entropy
python3 run_experiments.py -n 64 -b 1 -k 2 --folder twenty_sparse --strategy random
python3 run_experiments.py -n 64 -b 2 -k 2 --folder twenty_sparse --strategy entropy
python3 run_experiments.py -n 64 -b 2 -k 2 --folder twenty_sparse --strategy random
python3 run_experiments.py -n 64 -b 4 -k 2 --folder twenty_sparse --strategy entropy
python3 run_experiments.py -n 64 -b 4 -k 2 --folder twenty_sparse --strategy random

python3 run_experiments.py -n 16 -b 1 -k 4 --folder twenty_sparse --strategy entropy
python3 run_experiments.py -n 16 -b 1 -k 4 --folder twenty_sparse --strategy random
python3 run_experiments.py -n 16 -b 2 -k 4 --folder twenty_sparse --strategy entropy
python3 run_experiments.py -n 16 -b 2 -k 4 --folder twenty_sparse --strategy random
python3 run_experiments.py -n 16 -b 4 -k 4 --folder twenty_sparse --strategy entropy
python3 run_experiments.py -n 16 -b 4 -k 4 --folder twenty_sparse --strategy random

python3 run_experiments.py -n 64 -b 1 -k 4 --folder twenty_sparse --strategy entropy
python3 run_experiments.py -n 64 -b 1 -k 4 --folder twenty_sparse --strategy random
python3 run_experiments.py -n 64 -b 2 -k 4 --folder twenty_sparse --strategy entropy
python3 run_experiments.py -n 64 -b 2 -k 4 --folder twenty_sparse --strategy random
python3 run_experiments.py -n 64 -b 4 -k 4 --folder twenty_sparse --strategy entropy
python3 run_experiments.py -n 64 -b 4 -k 4 --folder twenty_sparse --strategy random

python3 get_parent_probs.py --folder twenty_sparse --target 9
tar -czf data/twenty_sparse.tar.gz data/twenty_sparse

