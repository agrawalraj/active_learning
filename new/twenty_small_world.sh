#!/usr/bin/env bash

python3 make_dataset.py -p 20 -s .5 -d 50 -t small_world --folder twenty_small_world
python3 run_experiments.py -n 24 -b 2 -k 2 --folder twenty_small_world --strategy random
python3 run_experiments.py -n 24 -b 2 -k 2 --folder twenty_small_world --strategy edge-prob
python3 run_experiments.py -n 24 -b 2 -k 2 --folder twenty_small_world --strategy learn-parents
python3 run_experiments.py -n 24 -b 3 -k 2 --folder twenty_small_world --strategy random
python3 run_experiments.py -n 24 -b 3 -k 2 --folder twenty_small_world --strategy edge-prob
python3 run_experiments.py -n 24 -b 3 -k 2 --folder twenty_small_world --strategy learn-parents
python3 run_experiments.py -n 24 -b 4 -k 2 --folder twenty_small_world --strategy random
python3 run_experiments.py -n 24 -b 4 -k 2 --folder twenty_small_world --strategy edge-prob
python3 run_experiments.py -n 24 -b 4 -k 2 --folder twenty_small_world --strategy learn-parents

python3 run_experiments.py -n 96 -b 2 -k 2 --folder twenty_small_world --strategy random
python3 run_experiments.py -n 96 -b 2 -k 2 --folder twenty_small_world --strategy edge-prob
python3 run_experiments.py -n 96 -b 2 -k 2 --folder twenty_small_world --strategy learn-parents
python3 run_experiments.py -n 96 -b 3 -k 2 --folder twenty_small_world --strategy random
python3 run_experiments.py -n 96 -b 3 -k 2 --folder twenty_small_world --strategy edge-prob
python3 run_experiments.py -n 96 -b 3 -k 2 --folder twenty_small_world --strategy learn-parents
python3 run_experiments.py -n 96 -b 4 -k 2 --folder twenty_small_world --strategy random
python3 run_experiments.py -n 96 -b 4 -k 2 --folder twenty_small_world --strategy edge-prob
python3 run_experiments.py -n 96 -b 4 -k 2 --folder twenty_small_world --strategy learn-parents

python3 run_experiments.py -n 384 -b 2 -k 2 --folder twenty_small_world --strategy random
python3 run_experiments.py -n 384 -b 2 -k 2 --folder twenty_small_world --strategy edge-prob
python3 run_experiments.py -n 384 -b 2 -k 2 --folder twenty_small_world --strategy learn-parents
python3 run_experiments.py -n 384 -b 3 -k 2 --folder twenty_small_world --strategy random
python3 run_experiments.py -n 384 -b 3 -k 2 --folder twenty_small_world --strategy edge-prob
python3 run_experiments.py -n 384 -b 3 -k 2 --folder twenty_small_world --strategy learn-parents
python3 run_experiments.py -n 384 -b 4 -k 2 --folder twenty_small_world --strategy random
python3 run_experiments.py -n 384 -b 4 -k 2 --folder twenty_small_world --strategy edge-prob
python3 run_experiments.py -n 384 -b 4 -k 2 --folder twenty_small_world --strategy learn-parents

tar -czf data/twenty_small_world.tar.gz data/twenty_small_world

