from __future__ import division  # in case python2 is used

import os
import shutil
import numpy as np
import networkx as nx
import itertools as itr
from scipy.stats import multivariate_normal
from scipy.linalg import ldl
import operator as op
import config
import pandas as pd
import causaldag as cd


def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)


def RAND_RANGE():
    return np.random.uniform(.25, 1) * (-1 if bernoulli(.5) else 1)


def inv_perm(permutation):
    return [i for i, j in sorted(enumerate(permutation), key=op.itemgetter(1))]


def bernoulli(p):
    return np.random.binomial(1, p)


def permute(mat, perm):
    m = mat.copy()
    m = m[perm]
    m = m[:, perm]
    return m


def random_graph(p, edge_prob):
    g = nx.DiGraph()
    for i in range(p):
        g.add_node(i)
    for i, j in itr.combinations(range(p), 2):
        if bernoulli(edge_prob):
            g.add_edge(i, j)
    return g


def reverse_edge(g, xi, xj):
    # attrs = g[xi][xj]
    g.remove_edge(xi, xj)
    # g.add_edge(xj, xi, attrs)
    g.add_edge(xj, xi)


def get_target_parent_probs(target, adj_mats):
    num_nodes = adj_mats[0].shape[0]
    probs = np.array(num_nodes)
    avg_adj_mat = np.zeros(adj_mats[0].shape)
    for adj_mat in adj_mats:
        avg_adj_mat += adj_mat
    avg_adj_mat *= 1 / len(adj_mats)
    return(avg_adj_mat[target, :])


def random_adj(g, num_gen=RAND_RANGE):
    p = len(g.nodes)
    adj_mat = np.zeros([p, p])
    for i, j in g.edges:
        adj_mat[i, j] = num_gen()
    return adj_mat


def zero_rowcol(adj_mat, intervention):
    adj_mat_int = adj_mat.copy()
    adj_mat_int[intervention, :] = 0
    adj_mat_int[:, intervention] = 0
    return adj_mat_int


def adj2prec(adj_mat, omega=None):
    p = adj_mat.shape[0]
    i = np.eye(p)
    if omega is None:
        return (i - adj_mat) @ (i - adj_mat).T
    else:
        return (i - adj_mat) @ np.linalg.inv(omega) @ (i - adj_mat).T


def adj2cov(adj_mat, omega=None):
    p = adj_mat.shape[0]
    i = np.eye(p)
    a = np.linalg.inv(i - adj_mat)  # to avoid inverting twice
    if omega is None:
        return a.T @ a
    else:
        return a.T @ omega @ a


def adj2cov_int(adj_mat, intervention, omega=None):
    p = adj_mat.shape[0]
    i = np.eye(p)
    adj_mat_int = zero_rowcol(adj_mat, intervention)
    a = np.linalg.inv(i - adj_mat_int)  # to avoid inverting twice
    if omega is None:
        return a.T @ a
    else:
        omega_int = zero_rowcol(omega, intervention)
        return a.T @ omega_int @ a


def adj2inc(adj_mat):
    inc_mat = adj_mat.copy()
    inc_mat[inc_mat != 0] = 1
    return(inc_mat)


def prec2adj(prec, node_order):
    if not is_pos_def(prec):
        raise ValueError('precision matrix is not positive definite')
    p = prec.shape[0]
    prec = permute(prec, node_order)
    u, d, perm_ = ldl(prec, lower=False)
    u[np.isclose(u, 0)] = 0
    inv_node_order = inv_perm(node_order)
    adj_mat = np.eye(p) - permute(u, inv_node_order)
    omega = np.linalg.inv(permute(d, inv_node_order))
    adj_mat[np.isclose(adj_mat, 0)] = 0
    return adj_mat, omega


def update_order(curr_order, i, j):
    new_order = []
    for k in curr_order:
        if k == i:
            new_order.append(j)
            new_order.append(i)
        elif k == j:
            continue
        else:
            new_order.append(k)
    return new_order


def sample_graph_obs(cov_mat, n_samples):
    p = cov_mat.shape[0]
    return np.random.multivariate_normal(np.zeros(p), cov_mat, n_samples)


def sample_graph_int(g, adj_mat, interventions, n_samples):
    all_samples = [[] for _ in range(len(g.nodes))]
    for j, intervention in enumerate(interventions):
        adj_mat_int = zero_rowcol(adj_mat, intervention)
        cov_mat = adj2cov(adj_mat_int)
        all_samples[intervention] = sample_graph_obs(cov_mat, n_samples[j])
    return all_samples


def compute_log_posterior_unnormalized(g, node_order, siginv, int_data):
    log_post = 0
    adj_mat, omega = prec2adj(siginv, node_order)
    for i in range(len(g.nodes)):
        data = int_data[i]
        if len(data) != 0:
            cov_mat_int = adj2cov_int(adj_mat, i)
            m = multivariate_normal.pdf(data, cov=cov_mat_int)
            log_post += np.sum(np.log(m))
    return log_post


def is_covered_edge(g, source, target):
    return set(g.pred[source]) == set(g.pred[target]) - {source}


def get_covered_edges(g):
    cov_edges = set()
    for source, target in g.edges:
        if is_covered_edge(g, source, target):
            cov_edges.add((source, target))
    return cov_edges


def update_covered_edges(g, xi, xj, prev_cov_edge_set):
    # Only children of nodes xi and xj need to be updated
    good_cov_edges = set([edge for edge in prev_cov_edge_set if edge[0] != xi and edge[0] != xj])

    # Update edges between xi and xj
    [good_cov_edges.add((xi, child)) for child in set(g[xi]) if is_covered_edge(g, xi, child)]
    [good_cov_edges.add((xj, child)) for child in set(g[xj]) if is_covered_edge(g, xj, child)]
    return good_cov_edges


def run_min_imap(data_path, intervention_path, alpha=.05, gamma=1, 
    n_iter=50000, save_step=100, path=config.TEMP_DAG_FOLDER, delete=False):
    # delete all DAGS in TEMP FOLDER
    if delete:
        try:
            shutil.rmtree(path)
            os.mkdir(path)
            print('All files deleted in ' + path)
        except Exception as e:
            os.mkdir(path)
            print('Made TEMP DAG directory')
    rfile = os.path.join(config.TOP_FOLDER, 'utils', 'minIMAP.r')
    r_command = 'Rscript {} {} {} {} {} {} {} {}'.format(rfile, data_path, intervention_path,
        str(alpha), str(gamma), str(n_iter), str(save_step), path)
    os.system(r_command)


def run_gies_boot(n_boot, data_path, intervention_path, path=config.TEMP_DAG_FOLDER, delete=False):
    # delete all DAGS in TEMP FOLDER
    if delete:
        try:
            shutil.rmtree(path)
            os.mkdir(path)
            print('All files deleted in ' + path)
        except Exception as e:
            os.mkdir(path)
            print('Made TEMP DAG directory')
    rfile = os.path.join(config.TOP_FOLDER, 'utils', 'run_gies.r')
    r_command = 'Rscript {} {} {} {} {}'.format(rfile, n_boot, data_path, intervention_path, path)
    os.system(r_command)


def _write_data(data):
    """
    Helper function to write interventional data to files so that it can be used by R
    """
    # clear current data
    open(config.TEMP_SAMPLES_PATH, 'w').close()
    open(config.TEMP_INTERVENTIONS_PATH, 'w').close()

    iv_nodes = []
    for iv_node, samples in data.items():
        with open(config.TEMP_SAMPLES_PATH, 'ab') as f:
            np.savetxt(f, samples)
        iv_nodes.extend([iv_node+1 if iv_node != -1 else -1]*len(samples))
    pd.Series(iv_nodes).to_csv(config.TEMP_INTERVENTIONS_PATH, index=False)


def _load_dags():
    """
    Helper function to load the DAGs generated in R
    """
    adj_mats = []
    paths = os.listdir(config.TEMP_DAG_FOLDER)
    for file_path in paths:
        if 'score' not in file_path and '.DS_Store' not in file_path:
            adj_mat = pd.read_csv(os.path.join(config.TEMP_DAG_FOLDER, file_path))
            adj_mats.append(adj_mat.as_matrix())
    return [cd.DAG.from_amat(adj) for adj in adj_mats]