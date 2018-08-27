#!/usr/bin/env bash

python3 make_dataset.py -p 20 -s .8 -d 50 --folder twenty_dense
python3 run_experiments.py -n 24 -b 2 -k 2 --folder twenty_dense --strategy random
python3 run_experiments.py -n 24 -b 2 -k 2 --folder twenty_dense --strategy edge-prob
python3 run_experiments.py -n 24 -b 2 -k 2 --folder twenty_dense --strategy learn-parents
python3 run_experiments.py -n 24 -b 3 -k 2 --folder twenty_dense --strategy random
python3 run_experiments.py -n 24 -b 3 -k 2 --folder twenty_dense --strategy edge-prob
python3 run_experiments.py -n 24 -b 3 -k 2 --folder twenty_dense --strategy learn-parents
python3 run_experiments.py -n 24 -b 4 -k 2 --folder twenty_dense --strategy random
python3 run_experiments.py -n 24 -b 4 -k 2 --folder twenty_dense --strategy edge-prob
python3 run_experiments.py -n 24 -b 4 -k 2 --folder twenty_dense --strategy learn-parents

python3 run_experiments.py -n 60 -b 2 -k 2 --folder twenty_dense --strategy random
python3 run_experiments.py -n 60 -b 2 -k 2 --folder twenty_dense --strategy edge-prob
python3 run_experiments.py -n 60 -b 2 -k 2 --folder twenty_dense --strategy learn-parents
python3 run_experiments.py -n 60 -b 3 -k 2 --folder twenty_dense --strategy random
python3 run_experiments.py -n 60 -b 3 -k 2 --folder twenty_dense --strategy edge-prob
python3 run_experiments.py -n 60 -b 3 -k 2 --folder twenty_dense --strategy learn-parents
python3 run_experiments.py -n 60 -b 4 -k 2 --folder twenty_dense --strategy random
python3 run_experiments.py -n 60 -b 4 -k 2 --folder twenty_dense --strategy edge-prob
python3 run_experiments.py -n 60 -b 4 -k 2 --folder twenty_dense --strategy learn-parents

python3 run_experiments.py -n 120 -b 2 -k 2 --folder twenty_dense --strategy random
python3 run_experiments.py -n 120 -b 2 -k 2 --folder twenty_dense --strategy edge-prob
python3 run_experiments.py -n 120 -b 2 -k 2 --folder twenty_dense --strategy learn-parents
python3 run_experiments.py -n 120 -b 3 -k 2 --folder twenty_dense --strategy random
python3 run_experiments.py -n 120 -b 3 -k 2 --folder twenty_dense --strategy edge-prob
python3 run_experiments.py -n 120 -b 3 -k 2 --folder twenty_dense --strategy learn-parents
python3 run_experiments.py -n 120 -b 4 -k 2 --folder twenty_dense --strategy random
python3 run_experiments.py -n 120 -b 4 -k 2 --folder twenty_dense --strategy edge-prob
python3 run_experiments.py -n 120 -b 4 -k 2 --folder twenty_dense --strategy learn-parents

python3 run_experiments.py -n 240 -b 2 -k 2 --folder twenty_dense --strategy random
python3 run_experiments.py -n 240 -b 2 -k 2 --folder twenty_dense --strategy edge-prob
python3 run_experiments.py -n 240 -b 2 -k 2 --folder twenty_dense --strategy learn-parents
python3 run_experiments.py -n 240 -b 3 -k 2 --folder twenty_dense --strategy random
python3 run_experiments.py -n 240 -b 3 -k 2 --folder twenty_dense --strategy edge-prob
python3 run_experiments.py -n 240 -b 3 -k 2 --folder twenty_dense --strategy learn-parents
python3 run_experiments.py -n 240 -b 4 -k 2 --folder twenty_dense --strategy random
python3 run_experiments.py -n 240 -b 4 -k 2 --folder twenty_dense --strategy edge-prob
python3 run_experiments.py -n 240 -b 4 -k 2 --folder twenty_dense --strategy learn-parents



