#!/usr/bin/env bash

python3 run_experiments.py -n 32 -b 8 --folder twenty_no_limit_weak --strategy entropy
python3 run_experiments.py -n 32 -b 8 --folder twenty_no_limit_weak --strategy random
python3 run_experiments.py -n 32 -b 16 --folder twenty_no_limit_weak --strategy entropy
python3 run_experiments.py -n 32 -b 16 --folder twenty_no_limit_weak --strategy random

python3 run_experiments.py -n 64 -b 8 --folder twenty_no_limit_weak --strategy entropy
python3 run_experiments.py -n 64 -b 8 --folder twenty_no_limit_weak --strategy random
python3 run_experiments.py -n 64 -b 16 --folder twenty_no_limit_weak --strategy entropy
python3 run_experiments.py -n 64 -b 16 --folder twenty_no_limit_weak --strategy random

python3 get_parent_probs.py --folder twenty_no_limit_weak --target 9
tar -czf data/twenty_no_limit_weak.tar.gz data/twenty_no_limit_weak

