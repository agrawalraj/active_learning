#!/usr/bin/env bash

python3 make_dataset.py -p 20 -s .5 -d 50 -t erdos --folder twenty
python3 run_experiments.py -n 24 -b 2 -k 2 --folder twenty --strategy random
python3 run_experiments.py -n 24 -b 2 -k 2 --folder twenty --strategy var-score
python3 run_experiments.py -n 24 -b 3 -k 2 --folder twenty --strategy random
python3 run_experiments.py -n 24 -b 3 -k 2 --folder twenty --strategy var-score
python3 run_experiments.py -n 24 -b 4 -k 2 --folder twenty --strategy random
python3 run_experiments.py -n 24 -b 4 -k 2 --folder twenty --strategy var-score

python3 run_experiments.py -n 30 -b 2 -k 2 --folder twenty --strategy random
python3 run_experiments.py -n 30 -b 2 -k 2 --folder twenty --strategy var-score
python3 run_experiments.py -n 30 -b 3 -k 2 --folder twenty --strategy random
python3 run_experiments.py -n 30 -b 3 -k 2 --folder twenty --strategy var-score
python3 run_experiments.py -n 30 -b 4 -k 2 --folder twenty --strategy random
python3 run_experiments.py -n 30 -b 4 -k 2 --folder twenty --strategy var-score

python3 run_experiments.py -n 60 -b 2 -k 2 --folder twenty --strategy random
python3 run_experiments.py -n 60 -b 2 -k 2 --folder twenty --strategy var-score
python3 run_experiments.py -n 60 -b 3 -k 2 --folder twenty --strategy random
python3 run_experiments.py -n 60 -b 3 -k 2 --folder twenty --strategy var-score
python3 run_experiments.py -n 60 -b 4 -k 2 --folder twenty --strategy random
python3 run_experiments.py -n 60 -b 4 -k 2 --folder twenty --strategy var-score

python3 run_experiments.py -n 120 -b 2 -k 2 --folder twenty --strategy random
python3 run_experiments.py -n 120 -b 2 -k 2 --folder twenty --strategy var-score
python3 run_experiments.py -n 120 -b 3 -k 2 --folder twenty --strategy random
python3 run_experiments.py -n 120 -b 3 -k 2 --folder twenty --strategy var-score
python3 run_experiments.py -n 120 -b 4 -k 2 --folder twenty --strategy random
python3 run_experiments.py -n 120 -b 4 -k 2 --folder twenty --strategy var-score

python3 run_experiments.py -n 240 -b 2 -k 2 --folder twenty --strategy random
python3 run_experiments.py -n 240 -b 2 -k 2 --folder twenty --strategy var-score
python3 run_experiments.py -n 240 -b 3 -k 2 --folder twenty --strategy random
python3 run_experiments.py -n 240 -b 3 -k 2 --folder twenty --strategy var-score
python3 run_experiments.py -n 240 -b 4 -k 2 --folder twenty --strategy random
python3 run_experiments.py -n 240 -b 4 -k 2 --folder twenty --strategy var-score

tar -czf data/twenty.tar.gz data/twenty

python3