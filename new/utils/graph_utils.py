from __future__ import division  # in case python2 is used

import numpy as np
import networkx as nx
import itertools as itr
from scipy.stats import multivariate_normal
from scipy.linalg import ldl
import operator as op
from collections import Counter


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
    attrs = g[xi][xj]
    g.remove_edge(xi, xj)
    g.add_edge(xj, xi, attrs)


def show_graph(g, plt):
    nx.draw(g)
    plt.ion()
    plt.show()


def graph2adj(g):
    p = len(g.nodes)
    adj = np.zeros([p, p])
    for i, j in g.edges:
        adj[i, j] = 1
    return adj


def random_adj(g):
    p = len(g.nodes)
    adj_mat = np.zeros([p, p])
    for i, j in g.edges:
        adj_mat[i, j] = np.random.uniform(.25, 1)
        if bernoulli(.5):
            adj_mat[i, j] *= -1
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


def prec2adj(prec, node_order):
    p = prec.shape[0]
    prec = permute(prec, node_order)
    u, d, _ = ldl(prec, lower=False)
    inv_node_order = inv_perm(node_order)
    adj_mat = np.eye(p) - permute(u, inv_node_order)
    omega = np.linalg.inv(permute(d, inv_node_order))
    return adj_mat, omega


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


def concatenate_data(old_data, new_data):
    return [[*old_data[i], *new_data[i]] for i in range(len(old_data))]


def compute_log_posterior_unnormalized(g, siginv, int_data):
    log_post = 0
    adj_mat, omega = prec2adj(siginv, g.nodes)
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


def split_digraph(g):
    essgraph_dir = nx.DiGraph()
    essgraph_undir = nx.Graph()

    edge_counts = Counter()
    for i, j in g.edges:
        if (j, i) in edge_counts:
            edge_counts.update([(j, i)])
        else:
            edge_counts.update([(i, j)])

    for i, j in edge_counts:
        if edge_counts[(i, j)] == 2:
            i_, j_ = sorted([i, j])
            essgraph_undir.add_edge(i_, j_)
        else:
            essgraph_dir.add_edge(i, j)

    return essgraph_dir, essgraph_undir


def get_vstructures(g):
    protected_edges = set()

    for j in g.nodes:
        for i, k in itr.combinations(g.predecessors(j), 2):
            if not g.has_edge(i, k) and not g.has_edge(k, i):
                protected_edges.add((i, j))
                protected_edges.add((k, j))
    return protected_edges


def replace_unprotected(g, u=None, verbose=False):
    PROTECTED = 'P'
    UNDECIDED = 'U'
    NOT_PROTECTED = 'N'

    if u is None:
        u = nx.Graph()
        u.add_nodes_from(g.nodes)
    else:
        u = u.copy()

    d = g.copy()
    protected_edges = get_vstructures(g)
    undecided_edges = set(d.edges) - protected_edges
    edge_flags = {(i, j): PROTECTED for i, j in protected_edges}
    edge_flags.update({(i, j): UNDECIDED for i, j in undecided_edges})

    is_parent = lambda i, j: d.has_edge(i, j)
    is_adjacent = lambda i, j: d.has_edge(i, j) or d.has_edge(j, i) or u.has_edge(i, j)
    is_neighbor = lambda i, j: u.has_edge(i, j)

    m = 0
    while undecided_edges:
        m += 1
        for i, j in undecided_edges:
            flag = NOT_PROTECTED

            # check configuration (a) -- causal chain
            for k in d.predecessors(i):
                if not is_adjacent(k, j):
                    if edge_flags[(k, i)] == PROTECTED:
                        flag = PROTECTED
                        if verbose: print('edge %s-%s protected by rule (a)' % (i, j))
                        break
                    else:
                        if verbose: print('edge %s-%s undecided by rule (a)' % (i, j))
                        flag = UNDECIDED

            # check configuration (c) -- acyclicity
            if flag != PROTECTED:
                for k in d.predecessors(j):
                    if is_parent(i, k):
                        if edge_flags[(i, k)] == PROTECTED and edge_flags[(k, j)] == PROTECTED:
                            flag = PROTECTED
                            if verbose: print('edge %s-%s protected by rule (c)' % (i, j))
                            break
                        else:
                            print('edge %s-%s undecided by rule (c)' % (i, j))
                            flag = UNDECIDED

            # check configuration (d)
            if flag != PROTECTED:
                for k1, k2 in itr.combinations(d.predecessors(j), 2):
                    if is_neighbor(k2, i) and is_neighbor(k2, i) and not is_adjacent(k1, k2):
                        if edge_flags[(k1, j)] == PROTECTED and edge_flags[(k2, j)] == PROTECTED:
                            flag = PROTECTED
                            if verbose: print('edge %s-%s protected by rule (d)' % (i, j))
                            break
                        else:
                            if verbose: print('edge %s-%s undecided by rule (a)' % (i, j))
                            flag = UNDECIDED

            edge_flags[(i, j)] = flag

        # replace unprotected edges by lines
        for i, j in undecided_edges.copy():
            if edge_flags[(i, j)] != UNDECIDED:
                undecided_edges.remove((i, j))
            if edge_flags[(i, j)] == NOT_PROTECTED:
                u.add_edge(i, j)
                d.remove_edge(i, j)

    return d, u


def get_essgraph(g, verbose=False):
    return replace_unprotected(g, verbose=verbose)


def get_iessgraph(essgraph, interventions):
    iessgraph_dir = nx.DiGraph()
    iessgraph_undir = nx.Graph()
    # TODO
    return iessgraph_dir, iessgraph_undir


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    g = random_graph(10, .5)
    # adj_mat = random_adj(g)
    # omega = np.random.uniform(.5, 1, 10)
    # omega = np.diag(omega)
    # siginv = adj2prec(adj_mat, omega)
    # adj_mat2, omega2 = prec2adj(siginv, range(10))
    # l, d, _ = ldl(siginv)
    # print(np.allclose(adj_mat, adj_mat2))
    # print(np.allclose(omega, omega2))
    #
    # print(get_covered_edges(g))
    # int_data = sample_graph_int(g, adj_mat, [2, 4, 5, 9], [5]*4)
    # log_post = compute_log_posterior_unnormalized(g, siginv, int_data)

    # d, u = get_essgraph(g)
    # print(list(d.edges))
    # print(list(u.edges))
    # print(set(d.edges) | set(u.edges) == set(g.edges))

    # g = nx.DiGraph()
    # g.add_nodes_from(range(3))
    # g.add_edges_from([(3, 1), (1, 2)])
    # print(get_vstructures(g))
    # get_essgraph(g)
    #
    # g2 = nx.DiGraph()
    # g2.add_nodes_from(range(3))
    # g2.add_edges_from([(3, 2), (1, 2)])
    # print(get_vstructures(g2))
    # get_essgraph(g2)

    # g2 = nx.DiGraph()
    # g2.add_nodes_from(range(4))
    # g2.add_edges_from([
    #     (0, 3),
    #     (1, 3),
    #     (2, 3),
    #     (2, 4),
    #     (3, 4)
    # ])
    # d, u = get_essgraph(g2)
    # print(list(d.edges))
    # print(list(u.edges))

    g2 = nx.DiGraph()
    g2.add_nodes_from(range(4))
    g2.add_edges_from([
        (0, 1),
        (1, 2),
        (0, 3),
        (1, 2),
        (1, 4),
        (2, 3)
    ])
    d, u = get_essgraph(g2)
    print(list(d.edges))
    print(list(u.edges))





