import os
import itertools as itr

ns = [24, 48, 96, 192]
ks = [1]
bs = [1, 2, 3]

p = 25
s = .25
ndags = 50
strategies = {'random', 'entropy-multiple-mec'}

# os.system(f'python3 make_dataset.py -p {p} -s {.25} -d {ndags} -t erdos-bounded --folder erdos_renyi_multiple_mec')
for n, b, k, strat in itr.product(ns, bs, ks, strategies):
    if 'entropy-multiple-mec' == strat:
        if k is None:
            cmd = f'python3 run_experiments.py -n {n} -b {b} -m 2 -s .1 -i gauss --folder erdos_renyi_multiple_mec --strategy entropy-dag-collection-multiple-mec'
        else:
            cmd = f'python3 run_experiments.py -n {n} -b {b} -k {k} -m 2 -s .1 -i gauss --folder erdos_renyi_multiple_mec --strategy entropy-dag-collection-multiple-mec'
    if 'random' == strat:
        if k is None:
            cmd = f'python3 run_experiments.py -n {n} -b {b} -m 2 -s .1 -i gauss --folder erdos_renyi_multiple_mec --strategy random'
        else:
            cmd = f'python3 run_experiments.py -n {n} -b {b} -k {k} -m 2 -s .1 -i gauss --folder erdos_renyi_multiple_mec --strategy random'
    os.system(f'echo "{cmd}" > tmp.sh')
    os.system('cat slurm_template.sh tmp.sh > job.sh')
    os.system('rm tmp.sh')
    os.system('sbatch job.sh')
