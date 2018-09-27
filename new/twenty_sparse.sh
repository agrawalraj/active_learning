#!/usr/bin/env bash

python3 make_dataset.py -p 20 -s 1 -d 50 -t erdos --folder twenty_complete_new
python3 run_experiments.py -n 32 -b 1 -k 1 --boot 50 -s 2 --folder twenty_complete_new --strategy entropy
python3 run_experiments.py -n 32 -b 1 -k 1 --boot 50 -s 2 --folder twenty_complete_new --strategy random
python3 run_experiments.py -n 32 -b 2 -k 1 --boot 50 -s 2 --folder twenty_complete_new --strategy entropy
python3 run_experiments.py -n 32 -b 2 -k 1 --boot 50 -s 2 --folder twenty_complete_new --strategy random
python3 run_experiments.py -n 256 -b 1 -k 1 --boot 50 -s 2 --folder twenty_complete_new --strategy entropy
python3 run_experiments.py -n 256 -b 1 -k 1 --boot 50 -s 2 --folder twenty_complete_new --strategy random
python3 run_experiments.py -n 256 -b 2 -k 1 --boot 50 -s 2 --folder twenty_complete_new --strategy entropy
python3 run_experiments.py -n 256 -b 2 -k 1 --boot 50 -s 2 --folder twenty_complete_new --strategy random
python3 run_experiments.py -n 1024 -b 1 -k 1 --boot 50 -s 2 --folder twenty_complete_new --strategy entropy
python3 run_experiments.py -n 1024 -b 1 -k 1 --boot 50 -s 2 --folder twenty_complete_new --strategy random
python3 run_experiments.py -n 1024 -b 2 -k 1 --boot 50 -s 2 --folder twenty_complete_new --strategy entropy

python3 run_experiments.py -n 32 -b 1 -k 2 --boot 50 -s 2 --folder twenty_complete_new --strategy entropy
python3 run_experiments.py -n 32 -b 1 -k 2 --boot 50 -s 2 --folder twenty_complete_new --strategy random
python3 run_experiments.py -n 32 -b 2 -k 2 --boot 50 -s 2 --folder twenty_complete_new --strategy entropy
python3 run_experiments.py -n 32 -b 2 -k 2 --boot 50 -s 2 --folder twenty_complete_new --strategy random
python3 run_experiments.py -n 256 -b 1 -k 2 --boot 50 -s 2 --folder twenty_complete_new --strategy entropy
python3 run_experiments.py -n 256 -b 1 -k 2 --boot 50 -s 2 --folder twenty_complete_new --strategy random
python3 run_experiments.py -n 256 -b 2 -k 2 --boot 50 -s 2 --folder twenty_complete_new --strategy entropy
python3 run_experiments.py -n 256 -b 2 -k 2 --boot 50 -s 2 --folder twenty_complete_new --strategy random
python3 run_experiments.py -n 1024 -b 1 -k 2 --boot 50 -s 2 --folder twenty_complete_new --strategy entropy
python3 run_experiments.py -n 1024 -b 1 -k 2 --boot 50 -s 2 --folder twenty_complete_new --strategy random
python3 run_experiments.py -n 1024 -b 2 -k 2 --boot 50 -s 2 --folder twenty_complete_new --strategy entropy

python3 get_parent_probs.py --folder twenty_complete_new --target 9
tar -czf data/twenty_complete_new.tar.gz data/twenty_strong_complete_new