#!/usr/bin/env bash

python3 make_dataset.py -p 20 -s 1 -d 50 -t small_world --folder twenty_small_world_new
python3 run_experiments.py -n 32 -b 1 -k 1 --boot 100 -s 2 --folder twenty_small_world_new --strategy entropy-enum
python3 run_experiments.py -n 32 -b 2 -k 1 --boot 100 -s 2 --folder twenty_small_world_new --strategy entropy-enum
python3 run_experiments.py -n 256 -b 1 -k 1 --boot 100 -s 2 --folder twenty_small_world_new --strategy entropy-enum
python3 run_experiments.py -n 256 -b 2 -k 1 --boot 100 -s 2 --folder twenty_small_world_new --strategy entropy-enum
python3 run_experiments.py -n 32 -b 1 -k 2 --boot 100 -s 2 --folder twenty_small_world_new --strategy entropy-enum
python3 run_experiments.py -n 32 -b 2 -k 2 --boot 100 -s 2 --folder twenty_small_world_new --strategy entropy-enum
python3 run_experiments.py -n 256 -b 1 -k 2 --boot 100 -s 2 --folder twenty_small_world_new --strategy entropy-enum
python3 run_experiments.py -n 256 -b 2 -k 2 --boot 100 -s 2 --folder twenty_small_world_new --strategy entropy-enum

python3 run_experiments.py -n 32 -b 1 -k 1 --boot 100 -s 2 --folder twenty_small_world_new --strategy entropy
python3 run_experiments.py -n 32 -b 2 -k 1 --boot 100 -s 2 --folder twenty_small_world_new --strategy entropy
python3 run_experiments.py -n 256 -b 1 -k 1 --boot 100 -s 2 --folder twenty_small_world_new --strategy entropy
python3 run_experiments.py -n 256 -b 2 -k 1 --boot 100 -s 2 --folder twenty_small_world_new --strategy entropy
python3 run_experiments.py -n 32 -b 1 -k 2 --boot 100 -s 2 --folder twenty_small_world_new --strategy entropy
python3 run_experiments.py -n 32 -b 2 -k 2 --boot 100 -s 2 --folder twenty_small_world_new --strategy entropy
python3 run_experiments.py -n 256 -b 1 -k 2 --boot 100 -s 2 --folder twenty_small_world_new --strategy entropy
python3 run_experiments.py -n 256 -b 2 -k 2 --boot 100 -s 2 --folder twenty_small_world_new --strategy entropy

python3 run_experiments.py -n 32 -b 1 -k 1 --boot 100 -s 2 --folder twenty_small_world_new --strategy random
python3 run_experiments.py -n 32 -b 2 -k 1 --boot 100 -s 2 --folder twenty_small_world_new --strategy random
python3 run_experiments.py -n 256 -b 1 -k 1 --boot 100 -s 2 --folder twenty_small_world_new --strategy random
python3 run_experiments.py -n 256 -b 2 -k 1 --boot 100 -s 2 --folder twenty_small_world_new --strategy random
python3 run_experiments.py -n 32 -b 1 -k 2 --boot 100 -s 2 --folder twenty_small_world_new --strategy random
python3 run_experiments.py -n 32 -b 2 -k 2 --boot 100 -s 2 --folder twenty_small_world_new --strategy random
python3 run_experiments.py -n 256 -b 1 -k 2 --boot 100 -s 2 --folder twenty_small_world_new --strategy random
python3 run_experiments.py -n 256 -b 2 -k 2 --boot 100 -s 2 --folder twenty_small_world_new --strategy random

python3 get_parent_probs.py --folder twenty_small_world_new --target 9
tar -czf data/twenty_small_world_new.tar.gz data/twenty_small_world_new

python3 make_dataset.py -p 20 -s .5 -d 50 -t erdos --folder twenty_sparse_new
python3 run_experiments.py -n 32 -b 1 -k 1 --boot 100 -s 2 --folder twenty_sparse_new --strategy entropy-enum
python3 run_experiments.py -n 32 -b 2 -k 1 --boot 100 -s 2 --folder twenty_sparse_new --strategy entropy-enum
python3 run_experiments.py -n 256 -b 1 -k 1 --boot 100 -s 2 --folder twenty_sparse_new --strategy entropy-enum
python3 run_experiments.py -n 256 -b 2 -k 1 --boot 100 -s 2 --folder twenty_sparse_new --strategy entropy-enum
python3 run_experiments.py -n 32 -b 1 -k 2 --boot 100 -s 2 --folder twenty_sparse_new --strategy entropy-enum
python3 run_experiments.py -n 32 -b 2 -k 2 --boot 100 -s 2 --folder twenty_sparse_new --strategy entropy-enum
python3 run_experiments.py -n 256 -b 1 -k 2 --boot 100 -s 2 --folder twenty_sparse_new --strategy entropy-enum
python3 run_experiments.py -n 256 -b 2 -k 2 --boot 100 -s 2 --folder twenty_sparse_new --strategy entropy-enum

python3 run_experiments.py -n 32 -b 1 -k 1 --boot 100 -s 2 --folder twenty_sparse_new --strategy entropy
python3 run_experiments.py -n 32 -b 2 -k 1 --boot 100 -s 2 --folder twenty_sparse_new --strategy entropy
python3 run_experiments.py -n 256 -b 1 -k 1 --boot 100 -s 2 --folder twenty_sparse_new --strategy entropy
python3 run_experiments.py -n 256 -b 2 -k 1 --boot 100 -s 2 --folder twenty_sparse_new --strategy entropy
python3 run_experiments.py -n 32 -b 1 -k 2 --boot 100 -s 2 --folder twenty_sparse_new --strategy entropy
python3 run_experiments.py -n 32 -b 2 -k 2 --boot 100 -s 2 --folder twenty_sparse_new --strategy entropy
python3 run_experiments.py -n 256 -b 1 -k 2 --boot 100 -s 2 --folder twenty_sparse_new --strategy entropy
python3 run_experiments.py -n 256 -b 2 -k 2 --boot 100 -s 2 --folder twenty_sparse_new --strategy entropy

python3 run_experiments.py -n 32 -b 1 -k 1 --boot 100 -s 2 --folder twenty_sparse_new --strategy random
python3 run_experiments.py -n 32 -b 2 -k 1 --boot 100 -s 2 --folder twenty_sparse_new --strategy random
python3 run_experiments.py -n 256 -b 1 -k 1 --boot 100 -s 2 --folder twenty_sparse_new --strategy random
python3 run_experiments.py -n 256 -b 2 -k 1 --boot 100 -s 2 --folder twenty_sparse_new --strategy random
python3 run_experiments.py -n 32 -b 1 -k 2 --boot 100 -s 2 --folder twenty_sparse_new --strategy random
python3 run_experiments.py -n 32 -b 2 -k 2 --boot 100 -s 2 --folder twenty_sparse_new --strategy random
python3 run_experiments.py -n 256 -b 1 -k 2 --boot 100 -s 2 --folder twenty_sparse_new --strategy random
python3 run_experiments.py -n 256 -b 2 -k 2 --boot 100 -s 2 --folder twenty_sparse_new --strategy random

python3 get_parent_probs.py --folder twenty_sparse_new --target 9
tar -czf data/twenty_sparse_new.tar.gz data/twenty_sparse_new

python3 make_dataset.py -p 20 -s 1 -d 50 -t small_world --folder twenty_small_world_strong_new
python3 run_experiments.py -n 32 -b 1 -k 1 --boot 100 -s 5 --folder twenty_small_world_strong_new --strategy entropy-enum
python3 run_experiments.py -n 32 -b 2 -k 1 --boot 100 -s 5 --folder twenty_small_world_strong_new --strategy entropy-enum
python3 run_experiments.py -n 256 -b 1 -k 1 --boot 100 -s 5 --folder twenty_small_world_strong_new --strategy entropy-enum
python3 run_experiments.py -n 256 -b 2 -k 1 --boot 100 -s 5 --folder twenty_small_world_strong_new --strategy entropy-enum
python3 run_experiments.py -n 32 -b 1 -k 2 --boot 100 -s 5 --folder twenty_small_world_strong_new --strategy entropy-enum
python3 run_experiments.py -n 32 -b 2 -k 2 --boot 100 -s 5 --folder twenty_small_world_strong_new --strategy entropy-enum
python3 run_experiments.py -n 256 -b 1 -k 2 --boot 100 -s 5 --folder twenty_small_world_strong_new --strategy entropy-enum
python3 run_experiments.py -n 256 -b 2 -k 2 --boot 100 -s 5 --folder twenty_small_world_strong_new --strategy entropy-enum

python3 run_experiments.py -n 32 -b 1 -k 1 --boot 100 -s 5 --folder twenty_small_world_strong_new --strategy entropy
python3 run_experiments.py -n 32 -b 2 -k 1 --boot 100 -s 5 --folder twenty_small_world_strong_new --strategy entropy
python3 run_experiments.py -n 256 -b 1 -k 1 --boot 100 -s 5 --folder twenty_small_world_strong_new --strategy entropy
python3 run_experiments.py -n 256 -b 2 -k 1 --boot 100 -s 5 --folder twenty_small_world_strong_new --strategy entropy
python3 run_experiments.py -n 32 -b 1 -k 2 --boot 100 -s 5 --folder twenty_small_world_strong_new --strategy entropy
python3 run_experiments.py -n 32 -b 2 -k 2 --boot 100 -s 5 --folder twenty_small_world_strong_new --strategy entropy
python3 run_experiments.py -n 256 -b 1 -k 2 --boot 100 -s 5 --folder twenty_small_world_strong_new --strategy entropy
python3 run_experiments.py -n 256 -b 2 -k 2 --boot 100 -s 5 --folder twenty_small_world_strong_new --strategy entropy

python3 run_experiments.py -n 32 -b 1 -k 1 --boot 100 -s 5 --folder twenty_small_world_strong_new --strategy random
python3 run_experiments.py -n 32 -b 2 -k 1 --boot 100 -s 5 --folder twenty_small_world_strong_new --strategy random
python3 run_experiments.py -n 256 -b 1 -k 1 --boot 100 -s 5 --folder twenty_small_world_strong_new --strategy random
python3 run_experiments.py -n 256 -b 2 -k 1 --boot 100 -s 5 --folder twenty_small_world_strong_new --strategy random
python3 run_experiments.py -n 32 -b 1 -k 2 --boot 100 -s 5 --folder twenty_small_world_strong_new --strategy random
python3 run_experiments.py -n 32 -b 2 -k 2 --boot 100 -s 5 --folder twenty_small_world_strong_new --strategy random
python3 run_experiments.py -n 256 -b 1 -k 2 --boot 100 -s 5 --folder twenty_small_world_strong_new --strategy random
python3 run_experiments.py -n 256 -b 2 -k 2 --boot 100 -s 5 --folder twenty_small_world_strong_new --strategy random

python3 get_parent_probs.py --folder twenty_small_world_strong_new --target 9
tar -czf data/twenty_small_world_strong_new.tar.gz data/twenty_small_world_strong_new

python3 make_dataset.py -p 20 -s 1 -d 50 -t small_world --folder twenty_small_world_weak_new
python3 run_experiments.py -n 32 -b 1 -k 1 --boot 100 -s .5 --folder twenty_small_world_weak_new --strategy entropy-enum
python3 run_experiments.py -n 32 -b 2 -k 1 --boot 100 -s .5 --folder twenty_small_world_weak_new --strategy entropy-enum
python3 run_experiments.py -n 256 -b 1 -k 1 --boot 100 -s .5 --folder twenty_small_world_weak_new --strategy entropy-enum
python3 run_experiments.py -n 256 -b 2 -k 1 --boot 100 -s .5 --folder twenty_small_world_weak_new --strategy entropy-enum
python3 run_experiments.py -n 32 -b 1 -k 2 --boot 100 -s .5 --folder twenty_small_world_weak_new --strategy entropy-enum
python3 run_experiments.py -n 32 -b 2 -k 2 --boot 100 -s .5 --folder twenty_small_world_weak_new --strategy entropy-enum
python3 run_experiments.py -n 256 -b 1 -k 2 --boot 100 -s .5 --folder twenty_small_world_weak_new --strategy entropy-enum
python3 run_experiments.py -n 256 -b 2 -k 2 --boot 100 -s .5 --folder twenty_small_world_weak_new --strategy entropy-enum

python3 run_experiments.py -n 32 -b 1 -k 1 --boot 100 -s .5 --folder twenty_small_world_weak_new --strategy entropy
python3 run_experiments.py -n 32 -b 2 -k 1 --boot 100 -s .5 --folder twenty_small_world_weak_new --strategy entropy
python3 run_experiments.py -n 256 -b 1 -k 1 --boot 100 -s .5 --folder twenty_small_world_weak_new --strategy entropy
python3 run_experiments.py -n 256 -b 2 -k 1 --boot 100 -s .5 --folder twenty_small_world_weak_new --strategy entropy
python3 run_experiments.py -n 32 -b 1 -k 2 --boot 100 -s .5 --folder twenty_small_world_weak_new --strategy entropy
python3 run_experiments.py -n 32 -b 2 -k 2 --boot 100 -s .5 --folder twenty_small_world_weak_new --strategy entropy
python3 run_experiments.py -n 256 -b 1 -k 2 --boot 100 -s .5 --folder twenty_small_world_weak_new --strategy entropy
python3 run_experiments.py -n 256 -b 2 -k 2 --boot 100 -s .5 --folder twenty_small_world_weak_new --strategy entropy

python3 run_experiments.py -n 32 -b 1 -k 1 --boot 100 -s .5 --folder twenty_small_world_weak_new --strategy random
python3 run_experiments.py -n 32 -b 2 -k 1 --boot 100 -s .5 --folder twenty_small_world_weak_new --strategy random
python3 run_experiments.py -n 256 -b 1 -k 1 --boot 100 -s .5 --folder twenty_small_world_weak_new --strategy random
python3 run_experiments.py -n 256 -b 2 -k 1 --boot 100 -s .5 --folder twenty_small_world_weak_new --strategy random
python3 run_experiments.py -n 32 -b 1 -k 2 --boot 100 -s .5 --folder twenty_small_world_weak_new --strategy random
python3 run_experiments.py -n 32 -b 2 -k 2 --boot 100 -s .5 --folder twenty_small_world_weak_new --strategy random
python3 run_experiments.py -n 256 -b 1 -k 2 --boot 100 -s .5 --folder twenty_small_world_weak_new --strategy random
python3 run_experiments.py -n 256 -b 2 -k 2 --boot 100 -s .5 --folder twenty_small_world_weak_new --strategy random

python3 get_parent_probs.py --folder twenty_small_world_weak_new --target 9
tar -czf data/twenty_small_world_weak_new.tar.gz data/twenty_small_world_weak_new