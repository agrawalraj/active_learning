#!/usr/bin/env bash

python3 make_dataset.py -p 10 -s .5 -d 10 -t erdos --folder test2
python3 run_experiments.py -n 16 -b 1 -k 2 --folder test2 --strategy random