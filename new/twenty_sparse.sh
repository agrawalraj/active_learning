#!/usr/bin/env bash

python3 make_dataset.py -p 20 -s 1 -d 50 -t erdos --folder twenty_complete_new_info

python3 run_experiments.py -n 256 -b 1 -k 1 --boot 100 -s 2 --folder twenty_complete_new_info --strategy entropy
python3 run_experiments.py -n 256 -b 2 -k 1 --boot 100 -s 2 --folder twenty_complete_new_info --strategy entropy
python3 run_experiments.py -n 256 -b 1 -k 2 --boot 100 -s 2 --folder twenty_complete_new_info --strategy entropy
python3 run_experiments.py -n 256 -b 2 -k 2 --boot 100 -s 2 --folder twenty_complete_new_info --strategy entropy
python3 run_experiments.py -n 256 -b 1 --boot 100 -s 2 --folder twenty_complete_new_info --strategy entropy
python3 run_experiments.py -n 256 -b 2 --boot 100 -s 2 --folder twenty_complete_new_info --strategy entropy

python3 run_experiments.py -n 1024 -b 1 -k 1 --boot 100 -s 2 --folder twenty_complete_new_info --strategy entropy
python3 run_experiments.py -n 1024 -b 2 -k 1 --boot 100 -s 2 --folder twenty_complete_new_info --strategy entropy
python3 run_experiments.py -n 1024 -b 1 -k 2 --boot 100 -s 2 --folder twenty_complete_new_info --strategy entropy
python3 run_experiments.py -n 1024 -b 2 -k 2 --boot 100 -s 2 --folder twenty_complete_new_info --strategy entropy
python3 run_experiments.py -n 1024 -b 1 --boot 100 -s 2 --folder twenty_complete_new_info --strategy entropy
python3 run_experiments.py -n 1024 -b 2 --boot 100 -s 2 --folder twenty_complete_new_info --strategy entropy

python3 run_experiments.py -n 256 -b 1 -k 1 --boot 100 -s 2 --folder twenty_complete_new_info --strategy random
python3 run_experiments.py -n 256 -b 2 -k 1 --boot 100 -s 2 --folder twenty_complete_new_info --strategy random
python3 run_experiments.py -n 256 -b 1 -k 2 --boot 100 -s 2 --folder twenty_complete_new_info --strategy random
python3 run_experiments.py -n 256 -b 2 -k 2 --boot 100 -s 2 --folder twenty_complete_new_info --strategy random
python3 run_experiments.py -n 256 -b 1 --boot 100 -s 2 --folder twenty_complete_new_info --strategy random
python3 run_experiments.py -n 256 -b 2 --boot 100 -s 2 --folder twenty_complete_new_info --strategy random

python3 run_experiments.py -n 1024 -b 1 -k 1 --boot 100 -s 2 --folder twenty_complete_new_info --strategy random
python3 run_experiments.py -n 1024 -b 2 -k 1 --boot 100 -s 2 --folder twenty_complete_new_info --strategy random
python3 run_experiments.py -n 1024 -b 1 -k 2 --boot 100 -s 2 --folder twenty_complete_new_info --strategy random
python3 run_experiments.py -n 1024 -b 2 -k 2 --boot 100 -s 2 --folder twenty_complete_new_info --strategy random
python3 run_experiments.py -n 1024 -b 1 --boot 100 -s 2 --folder twenty_complete_new_info --strategy random
python3 run_experiments.py -n 1024 -b 2 --boot 100 -s 2 --folder twenty_complete_new_info --strategy random

python3 get_parent_probs.py --folder twenty_complete_new_info --target 9
tar -czf data/twenty_complete_new_info.tar.gz data/twenty_complete_new_info
