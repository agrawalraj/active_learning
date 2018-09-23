#!/usr/bin/env bash

python3 run_experiments.py -n 256 -b 1 -k 1 --folder twenty_complete --strategy entropy
python3 run_experiments.py -n 256 -b 1 -k 1 --folder twenty_complete --strategy random
python3 run_experiments.py -n 256 -b 2 -k 1 --folder twenty_complete --strategy entropy
python3 run_experiments.py -n 256 -b 2 -k 1 --folder twenty_complete --strategy random

python3 run_experiments.py -n 256 -b 1 -k 2 --folder twenty_complete --strategy entropy
python3 run_experiments.py -n 256 -b 1 -k 2 --folder twenty_complete --strategy random
python3 run_experiments.py -n 256 -b 2 -k 2 --folder twenty_complete --strategy entropy
python3 run_experiments.py -n 256 -b 2 -k 2 --folder twenty_complete --strategy random

python3 run_experiments.py -n 1024 -b 1 -k 1 --folder twenty_complete --strategy entropy
python3 run_experiments.py -n 1024 -b 1 -k 1 --folder twenty_complete --strategy random
python3 run_experiments.py -n 1024 -b 2 -k 1 --folder twenty_complete --strategy entropy
python3 run_experiments.py -n 1024 -b 2 -k 1 --folder twenty_complete --strategy random

python3 run_experiments.py -n 1024 -b 1 -k 2 --folder twenty_complete --strategy entropy
python3 run_experiments.py -n 1024 -b 1 -k 2 --folder twenty_complete --strategy random
python3 run_experiments.py -n 1024 -b 2 -k 2 --folder twenty_complete --strategy entropy
python3 run_experiments.py -n 1024 -b 2 -k 2 --folder twenty_complete --strategy random

python3 get_parent_probs.py --folder twenty_complete --target 9
tar -czf data/twenty_complete.tar.gz data/twenty_complete

