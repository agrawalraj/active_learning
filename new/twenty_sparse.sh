#!/usr/bin/env bash

python3 make_dataset.py -p 20 -s 1 -d 50 -t erdos --folder twenty_no_limit_weak
python3 run_experiments.py -n 64 -b 1 --folder twenty_no_limit_weak --strategy entropy
python3 run_experiments.py -n 64 -b 1 --folder twenty_no_limit_weak --strategy random
python3 run_experiments.py -n 64 -b 2 --folder twenty_no_limit_weak --strategy entropy
python3 run_experiments.py -n 64 -b 2 --folder twenty_no_limit_weak --strategy random
python3 run_experiments.py -n 64 -b 4 --folder twenty_no_limit_weak --strategy entropy
python3 run_experiments.py -n 64 -b 4 --folder twenty_no_limit_weak --strategy random

python3 run_experiments.py -n 256 -b 1 --folder twenty_no_limit_weak --strategy entropy
python3 run_experiments.py -n 256 -b 1 --folder twenty_no_limit_weak --strategy random
python3 run_experiments.py -n 256 -b 2 --folder twenty_no_limit_weak --strategy entropy
python3 run_experiments.py -n 256 -b 2 --folder twenty_no_limit_weak --strategy random
python3 run_experiments.py -n 256 -b 4 --folder twenty_no_limit_weak --strategy entropy
python3 run_experiments.py -n 256 -b 4 --folder twenty_no_limit_weak --strategy random

python3 run_experiments.py -n 1024 -b 1 --folder twenty_no_limit_weak --strategy entropy
python3 run_experiments.py -n 1024 -b 1 --folder twenty_no_limit_weak --strategy random
python3 run_experiments.py -n 1024 -b 2 --folder twenty_no_limit_weak --strategy entropy
python3 run_experiments.py -n 1024 -b 2 --folder twenty_no_limit_weak --strategy random
python3 run_experiments.py -n 1024 -b 4 --folder twenty_no_limit_weak --strategy entropy
python3 run_experiments.py -n 1024 -b 4 --folder twenty_no_limit_weak --strategy random

python3 get_parent_probs.py --folder twenty_no_limit_weak --target 9
tar -czf data/twenty_no_limit_weak.tar.gz data/twenty_no_limit_weak

