#!/usr/bin/env bash

python3 run_experiments.py -n 768 -b 2 -k 2 --folder fourty_small_world --strategy random
python3 run_experiments.py -n 768 -b 2 -k 2 --folder fourty_small_world --strategy edge-prob
python3 run_experiments.py -n 768 -b 2 -k 2 --folder fourty_small_world --strategy learn-parents
python3 run_experiments.py -n 768 -b 3 -k 2 --folder fourty_small_world --strategy random
python3 run_experiments.py -n 768 -b 3 -k 2 --folder fourty_small_world --strategy edge-prob
python3 run_experiments.py -n 768 -b 3 -k 2 --folder fourty_small_world --strategy learn-parents
python3 run_experiments.py -n 768 -b 4 -k 2 --folder fourty_small_world --strategy random
python3 run_experiments.py -n 768 -b 4 -k 2 --folder fourty_small_world --strategy edge-prob
python3 run_experiments.py -n 768 -b 4 -k 2 --folder fourty_small_world --strategy learn-parents

python3 run_experiments.py -n 1536 -b 2 -k 2 --folder fourty_small_world --strategy random
python3 run_experiments.py -n 1536 -b 2 -k 2 --folder fourty_small_world --strategy edge-prob
python3 run_experiments.py -n 1536 -b 2 -k 2 --folder fourty_small_world --strategy learn-parents
python3 run_experiments.py -n 1536 -b 3 -k 2 --folder fourty_small_world --strategy random
python3 run_experiments.py -n 1536 -b 3 -k 2 --folder fourty_small_world --strategy edge-prob
python3 run_experiments.py -n 1536 -b 3 -k 2 --folder fourty_small_world --strategy learn-parents
python3 run_experiments.py -n 1536 -b 4 -k 2 --folder fourty_small_world --strategy random
python3 run_experiments.py -n 1536 -b 4 -k 2 --folder fourty_small_world --strategy edge-prob
python3 run_experiments.py -n 1536 -b 4 -k 2 --folder fourty_small_world --strategy learn-parents

tar -czf data/fourty_small_world.tar.gz data/fourty_small_world

