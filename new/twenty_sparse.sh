#!/usr/bin/env bash

python3 make_dataset.py -p 20 -s 1 -d 50 -t erdos --folder twenty_complete
python3 run_experiments.py -n 16 -b 1 -k 1 --folder twenty_complete --strategy entropy
python3 run_experiments.py -n 16 -b 1 -k 1 --folder twenty_complete --strategy random
python3 run_experiments.py -n 16 -b 2 -k 1 --folder twenty_complete --strategy entropy
python3 run_experiments.py -n 16 -b 2 -k 1 --folder twenty_complete --strategy random
python3 run_experiments.py -n 16 -b 4 -k 1 --folder twenty_complete --strategy entropy
python3 run_experiments.py -n 16 -b 4 -k 1 --folder twenty_complete --strategy random

python3 run_experiments.py -n 64 -b 1 -k 1 --folder twenty_complete --strategy entropy
python3 run_experiments.py -n 64 -b 1 -k 1 --folder twenty_complete --strategy random
python3 run_experiments.py -n 64 -b 2 -k 1 --folder twenty_complete --strategy entropy
python3 run_experiments.py -n 64 -b 2 -k 1 --folder twenty_complete --strategy random
python3 run_experiments.py -n 64 -b 4 -k 1 --folder twenty_complete --strategy entropy
python3 run_experiments.py -n 64 -b 4 -k 1 --folder twenty_complete --strategy random

python3 run_experiments.py -n 16 -b 1 -k 2 --folder twenty_complete --strategy entropy
python3 run_experiments.py -n 16 -b 1 -k 2 --folder twenty_complete --strategy random
python3 run_experiments.py -n 16 -b 2 -k 2 --folder twenty_complete --strategy entropy
python3 run_experiments.py -n 16 -b 2 -k 2 --folder twenty_complete --strategy random
python3 run_experiments.py -n 16 -b 4 -k 2 --folder twenty_complete --strategy entropy
python3 run_experiments.py -n 16 -b 4 -k 2 --folder twenty_complete --strategy random

python3 run_experiments.py -n 64 -b 1 -k 2 --folder twenty_complete --strategy entropy
python3 run_experiments.py -n 64 -b 1 -k 2 --folder twenty_complete --strategy random
python3 run_experiments.py -n 64 -b 2 -k 2 --folder twenty_complete --strategy entropy
python3 run_experiments.py -n 64 -b 2 -k 2 --folder twenty_complete --strategy random
python3 run_experiments.py -n 64 -b 4 -k 2 --folder twenty_complete --strategy entropy
python3 run_experiments.py -n 64 -b 4 -k 2 --folder twenty_complete --strategy random

python3 get_parent_probs.py --folder twenty_complete --target 9
tar -czf data/twenty_complete.tar.gz data/twenty_complete

