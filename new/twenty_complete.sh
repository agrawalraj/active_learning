#!/usr/bin/env bash

python3 make_dataset.py -p 20 -s 1 -d 50 -t erdos --folder twenty_complete
python3 run_experiments.py -n 24 -b 2 -k 2 --folder twenty_complete --strategy random
python3 run_experiments.py -n 24 -b 2 -k 2 --folder twenty_complete --strategy var-score
python3 run_experiments.py -n 24 -b 2 -k 2 --folder twenty_complete --strategy edge-prob
python3 run_experiments.py -n 24 -b 3 -k 2 --folder twenty_complete --strategy random
python3 run_experiments.py -n 24 -b 3 -k 2 --folder twenty_complete --strategy var-score
python3 run_experiments.py -n 24 -b 3 -k 2 --folder twenty_complete --strategy edge-prob
python3 run_experiments.py -n 24 -b 4 -k 2 --folder twenty_complete --strategy random
python3 run_experiments.py -n 24 -b 4 -k 2 --folder twenty_complete --strategy var-score
python3 run_experiments.py -n 24 -b 4 -k 2 --folder twenty_complete --strategy edge-prob

python3 run_experiments.py -n 96 -b 2 -k 2 --folder twenty_complete --strategy random
python3 run_experiments.py -n 96 -b 2 -k 2 --folder twenty_complete --strategy var-score
python3 run_experiments.py -n 96 -b 2 -k 2 --folder twenty_complete --strategy edge-prob
python3 run_experiments.py -n 96 -b 3 -k 2 --folder twenty_complete --strategy random
python3 run_experiments.py -n 96 -b 3 -k 2 --folder twenty_complete --strategy var-score
python3 run_experiments.py -n 96 -b 3 -k 2 --folder twenty_complete --strategy edge-prob
python3 run_experiments.py -n 96 -b 4 -k 2 --folder twenty_complete --strategy random
python3 run_experiments.py -n 96 -b 4 -k 2 --folder twenty_complete --strategy var-score
python3 run_experiments.py -n 96 -b 4 -k 2 --folder twenty_complete --strategy edge-prob

python3 run_experiments.py -n 384 -b 2 -k 2 --folder twenty_complete --strategy random
python3 run_experiments.py -n 384 -b 2 -k 2 --folder twenty_complete --strategy var-score
python3 run_experiments.py -n 384 -b 2 -k 2 --folder twenty_complete --strategy edge-prob
python3 run_experiments.py -n 384 -b 3 -k 2 --folder twenty_complete --strategy random
python3 run_experiments.py -n 384 -b 3 -k 2 --folder twenty_complete --strategy var-score
python3 run_experiments.py -n 384 -b 3 -k 2 --folder twenty_complete --strategy edge-prob
python3 run_experiments.py -n 384 -b 4 -k 2 --folder twenty_complete --strategy random
python3 run_experiments.py -n 384 -b 4 -k 2 --folder twenty_complete --strategy var-score
python3 run_experiments.py -n 384 -b 4 -k 2 --folder twenty_complete --strategy edge-prob

python3 get_parent_probs.py --folder twenty_complete --target 10
tar -czf data/twenty_complete.tar.gz data/twenty_complete

