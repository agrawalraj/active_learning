import os
import itertools as itr

ns = [24, 48, 96, 192]
ks = [1]
bs = [1, 2, 3]

p = 25
s = .25
ndags = 50
strategies = {'random-smart'}

# os.system(f'python3 make_dataset.py -p {p} -s {.25} -d {ndags} -t erdos-bounded --folder erdos_renyi_final')
for n, b, k, strat in itr.product(ns, bs, ks, strategies):
    if 'bed' == strat:
        if k is None:
            cmd = f'python3 run_experiments.py -n {n} -b {b} -m 2 -s .1 -i gauss --folder erdos_renyi_final --strategy budgeted_exp_design'
        else:
            cmd = f'python3 run_experiments.py -n {n} -b 1 -k {b*k} -m 2 -s .1 -i gauss --folder erdos_renyi_final --strategy budgeted_exp_design'
    if 'entropy' == strat:
        if k is None:
            cmd = f'python3 run_experiments.py -n {n} -b {b} -m 2 -s .1 -i gauss --folder erdos_renyi_final --strategy entropy-dag-collection'
        else:
            cmd = f'python3 run_experiments.py -n {n} -b {b} -k {k} -m 2 -s .1 -i gauss --folder erdos_renyi_final --strategy entropy-dag-collection'
    if 'random' == strat:
        if k is None:
            cmd = f'python3 run_experiments.py -n {n} -b {b} -m 2 -s .1 -i gauss --folder erdos_renyi_final --strategy random'
        else:
            cmd = f'python3 run_experiments.py -n {n} -b {b} -k {k} -m 2 -s .1 -i gauss --folder erdos_renyi_final --strategy random'
    if 'random-smart' == strat:
        if k is None:
            cmd = f'python3 run_experiments.py -n {n} -b {b} -m 2 -s .1 -i gauss --folder erdos_renyi_final --strategy random-smart'
        else:
            cmd = f'python3 run_experiments.py -n {n} -b {b} -k {k} -m 2 -s .1 -i gauss --folder erdos_renyi_final --strategy random-smart'
    os.system(f'echo "{cmd}" > tmp.sh')
    os.system('cat slurm_template.sh tmp.sh > job.sh')
    os.system('rm tmp.sh')
    os.system('sbatch job.sh')
