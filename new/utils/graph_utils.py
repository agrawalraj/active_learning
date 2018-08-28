from __future__ import division  # in case python2 is used

import os
import shutil
import numpy as np
import config
import pandas as pd
import causaldag as cd
import networkx as nx
from networkx.utils import powerlaw_sequence


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


def generate_DAG(p, m=5, prob=.4, type_='config_model'):
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


