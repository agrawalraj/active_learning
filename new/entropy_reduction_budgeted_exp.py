from config import DATA_FOLDER
import itertools as itr
import numpy as np
import os
import causaldag as cd

ndags = 3
folder = 'erdos_renyi_final'
strategy = 'budgeted_exp_design'
dag_folders = [os.path.join(DATA_FOLDER, folder, 'dags', 'dag%d' % i) for i in range(ndags)]
dag_amats = [np.loadtxt(os.path.join(dag_folder, 'adjacency.txt')) for dag_folder in dag_folders]
dags = [cd.GaussDAG.from_amat(dag_amat) for dag_amat in dag_amats]
mecs = [[cd.DAG(arcs=arcs) for arcs in dag.cpdag().all_dags()] for dag in dags]

ns = [24, 48, 96]
bs = [1]
ks = [1, 2, 3, 4, 6, 9]
for n, b, k in itr.product(ns, bs, ks):
    folders = [os.path.join(DATA_FOLDER, folder, 'dags', 'dag%d' % i, strategy + ',n=%s,b=%s,k=%s' % (n, b, k)) for i in
               range(ndags)]
    dag_amats = [np.loadtxt()]
