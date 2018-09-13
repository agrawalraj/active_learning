from __future__ import division  # in case python2 is used

import os
import shutil
import numpy as np
import config
import pandas as pd
import networkx as nx
from networkx.utils import powerlaw_sequence
from sksparse.cholmod import cholesky  # this has to be used instead of scipy's because it doesn't permute the matrix
from scipy import sparse
import operator as op
import causaldag as cd


def bernoulli(p):
    return np.random.binomial(1, p)


def RAND_RANGE():
    return np.random.uniform(.25, 1) * (-1 if bernoulli(.5) else 1)


# def run_min_imap(data_path, intervention_path, alpha=.05, gamma=1,
#     n_iter=50000, save_step=100, path=config.TEMP_DAG_FOLDER, delete=False):
#     # delete all DAGS in TEMP FOLDER
#     if delete:
#         try:
#             shutil.rmtree(path)
#             os.mkdir(path)
#             print('All files deleted in ' + path)
#         except Exception as e:
#             os.mkdir(path)
#             print('Made TEMP DAG directory')
#     rfile = os.path.join(config.TOP_FOLDER, 'utils', 'minIMAP.r')
#     r_command = 'Rscript {} {} {} {} {} {} {} {}'.format(rfile, data_path, intervention_path,
#         str(alpha), str(gamma), str(n_iter), str(save_step), path)
#     os.system(r_command)


def run_gies_boot(n_boot, data_path, intervention_path, dags_path, delete=False):
    # delete all DAGS in TEMP FOLDER
    if delete:
        try:
            shutil.rmtree(dags_path)
            print('All DAGs deleted in ' + dags_path)
        except FileNotFoundError as e:
            pass
    if not os.path.exists(dags_path):
        os.mkdir(dags_path)
    rfile = os.path.join(config.TOP_FOLDER, 'utils', 'run_gies.r')
    r_command = 'Rscript {} {} {} {} {}'.format(rfile, n_boot, data_path, intervention_path, dags_path)
    os.system(r_command)


def _write_data(data, samples_path, interventions_path):
    """
    Helper function to write interventional data to files so that it can be used by R
    """
    # clear current data
    open(samples_path, 'w').close()
    open(interventions_path, 'w').close()

    iv_nodes = []
    for iv_node, samples in data.items():
        with open(samples_path, 'ab') as f:
            np.savetxt(f, samples)
        iv_nodes.extend([iv_node+1 if iv_node != -1 else -1]*len(samples))
    pd.Series(iv_nodes).to_csv(interventions_path, index=False)


def generate_DAG(p, m=5, prob=.05, type_='config_model'):
    if type_ == 'config_model':
        z = [int(e) for e in powerlaw_sequence(p)]
        if np.sum(z) % 2 != 0:
            z[0] += 1
        G = nx.configuration_model(z)
    elif type_ == 'barabasi':
        G = nx.barabasi_albert_graph(p, m)
    elif type_ == 'small_world':
        G = nx.watts_strogatz_graph(p, m, prob)
    else: 
        raise Exception('Not a graph type') 
    G = nx.Graph(G)
    dag = cd.DAG(nodes=set(range(p)))
    for i, j in G.edges:
        if i != j:
            dag.add_arc(*sorted((i, j)))
    return dag


def _load_dags(dags_path, delete=True):
    """
    Helper function to load the DAGs generated in R
    """
    adj_mats = []
    paths = os.listdir(dags_path)
    for file_path in paths:
        if 'score' not in file_path and '.DS_Store' not in file_path:
            adj_mat = pd.read_csv(os.path.join(dags_path, file_path))
            adj_mats.append(adj_mat.values)
            if delete:
                os.remove(os.path.join(dags_path, file_path))
    return adj_mats, [cd.DAG.from_amat(adj) for adj in adj_mats]


def probability_shrinkage(prob):
    return 2 * min(1 - prob, prob)


def entropy_shrinkage(prob):
    if prob == 0 or prob == 1:
        return 0
    return (prob * np.log(prob) + (1 - prob) * np.log(1 - prob)) / np.log(2)  


def prec2dag(prec, node_order):
    p = prec.shape[0]

    # === permute precision matrix into correct order for LDL
    prec = prec.copy()
    rev_node_order = list(reversed(node_order))
    prec = prec[rev_node_order]
    prec = prec[:, rev_node_order]

    # === perform ldl decomposition and correct for floating point errors
    factor = cholesky(sparse.csc_matrix(prec))
    l, d = factor.L_D()
    l = l.todense()
    d = d.todense()

    # === permute back
    inv_rev_node_order = [i for i, j in sorted(enumerate(rev_node_order), key=op.itemgetter(1))]
    l = l.copy()
    l = l[inv_rev_node_order]
    l = l[:, inv_rev_node_order]
    d = d.copy()
    d = d[inv_rev_node_order]
    d = d[:, inv_rev_node_order]

    amat = np.eye(p) - l
    variances = np.diag(d) ** -1

    return cd.GaussDAG.from_amat(amat, variances=variances)


if __name__ == '__main__':
    amat1 = np.array([
        [0, 2, 3],
        [0, 0, 5],
        [0, 0, 0]
    ])
    g1 = cd.GaussDAG.from_amat(amat1)
    prec = g1.precision
    g1_ = prec2dag(prec, [0, 1, 2])
    print(g1_.to_amat())

    g2 = prec2dag(prec, [0, 2, 1])
    print(g2.to_amat())
    print(g2.variances)
    print(g2.precision == g1.precision)

    g3 = prec2dag(prec, [1, 0, 2])
    print(g3.to_amat())
    print(g3.variances)
    print(g3.precision == g1.precision)

    g4 = prec2dag(prec, [1, 2, 0])
    print(g4.to_amat())
    print(g4.variances)
    print(g4.precision == g1.precision)

    g5 = prec2dag(prec, [2, 0, 1])
    print(g5.to_amat())
    print(g5.variances)
    print(g5.precision == g1.precision)

    g6 = prec2dag(prec, [2, 1, 0])
    print(g6.to_amat())
    print(g6.variances)
    print(g6.precision == g1.precision)

