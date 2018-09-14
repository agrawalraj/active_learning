#!/usr/bin/env bash

python3 make_dataset.py -p 50 -s .1 -d 100 -t small_world --folder fifty_small_world

python3 run_experiments.py -n 24 -b 2 -k 2 --folder fifty_small_world --strategy random
python3 run_experiments.py -n 24 -b 2 -k 2 --folder fifty_small_world --strategy edge-prob
python3 run_experiments.py -n 24 -b 2 -k 2 --folder fifty_small_world --strategy learn-parents
python3 run_experiments.py -n 24 -b 3 -k 2 --folder fifty_small_world --strategy random
python3 run_experiments.py -n 24 -b 3 -k 2 --folder fifty_small_world --strategy edge-prob
python3 run_experiments.py -n 24 -b 3 -k 2 --folder fifty_small_world --strategy learn-parents
python3 run_experiments.py -n 24 -b 4 -k 2 --folder fifty_small_world --strategy random
python3 run_experiments.py -n 24 -b 4 -k 2 --folder fifty_small_world --strategy edge-prob
python3 run_experiments.py -n 24 -b 4 -k 2 --folder fifty_small_world --strategy learn-parents

python3 run_experiments.py -n 48 -b 2 -k 2 --folder fifty_small_world --strategy random
python3 run_experiments.py -n 48 -b 2 -k 2 --folder fifty_small_world --strategy edge-prob
python3 run_experiments.py -n 48 -b 2 -k 2 --folder fifty_small_world --strategy learn-parents
python3 run_experiments.py -n 48 -b 3 -k 2 --folder fifty_small_world --strategy random
python3 run_experiments.py -n 48 -b 3 -k 2 --folder fifty_small_world --strategy edge-prob
python3 run_experiments.py -n 48 -b 3 -k 2 --folder fifty_small_world --strategy learn-parents
python3 run_experiments.py -n 48 -b 4 -k 2 --folder fifty_small_world --strategy random
python3 run_experiments.py -n 48 -b 4 -k 2 --folder fifty_small_world --strategy edge-prob
python3 run_experiments.py -n 48 -b 4 -k 2 --folder fifty_small_world --strategy learn-parents

python3 run_experiments.py -n 96 -b 2 -k 2 --folder fifty_small_world --strategy random
python3 run_experiments.py -n 96 -b 2 -k 2 --folder fifty_small_world --strategy edge-prob
python3 run_experiments.py -n 96 -b 2 -k 2 --folder fifty_small_world --strategy learn-parents
python3 run_experiments.py -n 96 -b 3 -k 2 --folder fifty_small_world --strategy random
python3 run_experiments.py -n 96 -b 3 -k 2 --folder fifty_small_world --strategy edge-prob
python3 run_experiments.py -n 96 -b 3 -k 2 --folder fifty_small_world --strategy learn-parents
python3 run_experiments.py -n 96 -b 4 -k 2 --folder fifty_small_world --strategy random
python3 run_experiments.py -n 96 -b 4 -k 2 --folder fifty_small_world --strategy edge-prob
python3 run_experiments.py -n 96 -b 4 -k 2 --folder fifty_small_world --strategy learn-parents

python3 run_experiments.py -n 192 -b 2 -k 2 --folder fifty_small_world --strategy random
python3 run_experiments.py -n 192 -b 2 -k 2 --folder fifty_small_world --strategy edge-prob
python3 run_experiments.py -n 192 -b 2 -k 2 --folder fifty_small_world --strategy learn-parents
python3 run_experiments.py -n 192 -b 3 -k 2 --folder fifty_small_world --strategy random
python3 run_experiments.py -n 192 -b 3 -k 2 --folder fifty_small_world --strategy edge-prob
python3 run_experiments.py -n 192 -b 3 -k 2 --folder fifty_small_world --strategy learn-parents
python3 run_experiments.py -n 192 -b 4 -k 2 --folder fifty_small_world --strategy random
python3 run_experiments.py -n 192 -b 4 -k 2 --folder fifty_small_world --strategy edge-prob
python3 run_experiments.py -n 192 -b 4 -k 2 --folder fifty_small_world --strategy learn-parents

tar -czf data/fifty_small_world.tar.gz data/fifty_small_world

