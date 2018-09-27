#!/usr/bin/env bash

python3 make_dataset.py -p 20 -s .5 -d 50 -t erdos --folder twenty_no_limit_strong_sparse
python3 run_experiments.py -n 32 -b 8 --folder twenty_no_limit_strong_sparse --strategy entropy
python3 run_experiments.py -n 32 -b 8 --folder twenty_no_limit_strong_sparse --strategy random
python3 run_experiments.py -n 32 -b 16 --folder twenty_no_limit_strong_sparse --strategy entropy
python3 run_experiments.py -n 32 -b 16 --folder twenty_no_limit_strong_sparse --strategy random

python3 run_experiments.py -n 64 -b 8 --folder twenty_no_limit_strong_sparse --strategy entropy
python3 run_experiments.py -n 64 -b 8 --folder twenty_no_limit_strong_sparse --strategy random
python3 run_experiments.py -n 64 -b 16 --folder twenty_no_limit_strong_sparse --strategy entropy
python3 run_experiments.py -n 64 -b 16 --folder twenty_no_limit_strong_sparse --strategy random

python3 get_parent_probs.py --folder twenty_no_limit_strong_sparse --target 9
tar -czf data/twenty_no_limit_strong_sparse.tar.gz data/twenty_no_limit_strong_sparse

python3 make_dataset.py -p 10 -s .5 -d 50 -t erdos --folder ten_no_limit_strong_sparse
python3 run_experiments.py -n 32 -b 8 --folder ten_no_limit_strong_sparse --strategy entropy
python3 run_experiments.py -n 32 -b 8 --folder ten_no_limit_strong_sparse --strategy random
python3 run_experiments.py -n 32 -b 16 --folder ten_no_limit_strong_sparse --strategy entropy
python3 run_experiments.py -n 32 -b 16 --folder ten_no_limit_strong_sparse --strategy random

python3 run_experiments.py -n 64 -b 8 --folder ten_no_limit_strong_sparse --strategy entropy
python3 run_experiments.py -n 64 -b 8 --folder ten_no_limit_strong_sparse --strategy random
python3 run_experiments.py -n 64 -b 16 --folder ten_no_limit_strong_sparse --strategy entropy
python3 run_experiments.py -n 64 -b 16 --folder ten_no_limit_strong_sparse --strategy random

python3 get_parent_probs.py --folder ten_no_limit_strong_sparse --target 9
tar -czf data/ten_no_limit_strong_sparse.tar.gz data/ten_no_limit_strong_sparse

python3 make_dataset.py -p 10 -s 1 -d 50 -t erdos --folder ten_no_limit_strong
python3 run_experiments.py -n 32 -b 8 --folder ten_no_limit_strong --strategy entropy
python3 run_experiments.py -n 32 -b 8 --folder ten_no_limit_strong --strategy random
python3 run_experiments.py -n 32 -b 16 --folder ten_no_limit_strong --strategy entropy
python3 run_experiments.py -n 32 -b 16 --folder ten_no_limit_strong --strategy random

python3 run_experiments.py -n 64 -b 8 --folder ten_no_limit_strong --strategy entropy
python3 run_experiments.py -n 64 -b 8 --folder ten_no_limit_strong --strategy random
python3 run_experiments.py -n 64 -b 16 --folder ten_no_limit_strong --strategy entropy
python3 run_experiments.py -n 64 -b 16 --folder ten_no_limit_strong --strategy random

python3 get_parent_probs.py --folder ten_no_limit_strong --target 9
tar -czf data/ten_no_limit_strong.tar.gz data/ten_no_limit_strong
