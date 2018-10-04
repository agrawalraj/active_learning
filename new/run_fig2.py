import os
import itertools as itr

ns = [24, 48, 96, 192]
ks = [1, None]
bs = [1, 2, 3]

p = 25
s = .25
ndags = 50

os.system(f'python3 make_dataset.py -p {p} -s {.25} -d {ndags} -t erdos-bounded --folder erdos_renyi_final')
for n, b, k in itr.product(ns, bs, ks):
    os.system(f'python3 run_experiments.py -n {n} -b {b} -k {k} -m 2 -s .1 -i gauss --folder erdos_renyi_final --strategy entropy-dag-collection')
for n, b, k in itr.product(ns, bs, ks):
    os.system(f'python3 run_experiments.py -n {n} -b {b} -k {k} -m 2 -s .1 -i gauss --folder erdos_renyi_final --strategy random')
for n, b, k in itr.product(ns, bs, ks):
    os.system(f'python3 run_experiments.py -n {n} -b {b} -k {k} -m 2 -s .1 -i gauss --folder erdos_renyi_final --strategy random-smart')
