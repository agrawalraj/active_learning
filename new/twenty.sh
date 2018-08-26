#!/usr/bin/env bash

python3 run_experiments.py -n 60 -b 3 -k 2 --folder twenty --strategy random
python3 run_experiments.py -n 60 -b 3 -k 2 --folder twenty --strategy edge-prob
python3 run_experiments.py -n 60 -b 3 -k 2 --folder twenty --strategy learn-parents
python3 run_experiments.py -n 120 -b 3 -k 2 --folder twenty --strategy random
python3 run_experiments.py -n 120 -b 3 -k 2 --folder twenty --strategy edge-prob
python3 run_experiments.py -n 120 -b 3 -k 2 --folder twenty --strategy learn-parents
python3 run_experiments.py -n 60 -b 4 -k 2 --folder twenty --strategy random
python3 run_experiments.py -n 60 -b 4 -k 2 --folder twenty --strategy edge-prob
python3 run_experiments.py -n 60 -b 4 -k 2 --folder twenty --strategy learn-parents
python3 run_experiments.py -n 120 -b 4 -k 2 --folder twenty --strategy random
python3 run_experiments.py -n 120 -b 4 -k 2 --folder twenty --strategy edge-prob
python3 run_experiments.py -n 120 -b 4 -k 2 --folder twenty --strategy learn-parents