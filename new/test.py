import os
import itertools as itr

ns = [24, 48]
ks = [None]
bs = [1, 2]

p = 11
s = .25
ndags = 20

os.system(f'python3 make_dataset.py -p {p} -s {.25} -d {ndags} -t unoriented_by_one --folder erdos_renyi_descendant_new')
for n, b, k in itr.product(ns, bs, ks):
    if k is None:
        os.system(f'python3 run_experiments.py -n {n} -b {b} -m 2 -s .1 -i gauss --target-allowed 0 --folder erdos_renyi_descendant_new --strategy random-smart')
    else:
        os.system(f'python3 run_experiments.py -n {n} -b {b} -k {k} -m 2 -s .1 -i gauss --target-allowed 0 --folder erdos_renyi_descendant_new --strategy random-smart')
