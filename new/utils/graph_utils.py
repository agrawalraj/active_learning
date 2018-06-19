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


def show_graph(g, plt):
    nx.draw(g)
    plt.ion()
    plt.show()


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


def get_covered_edges(g):
    cov_edges = set()
    for source, target in g.edges:
        if set(g.pred[source]) == set(g.pred[target]) - {source}:
            cov_edges.add((source, target))
    return cov_edges


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


def get_protected_edges(g, interventions=None):
    protected_edges = set()
    for i, j in g.edges:
        i_neighbors = set(g.predecessors(i)) | set(g.successors(i))
        is_vstruct = len(set(g.predecessors(j)) - i_neighbors - {i}) > 0
        if is_vstruct:
            protected_edges.add((i, j))
    if interventions is not None:
        # TODO
        pass
    return protected_edges


def get_essgraph(g):
    d = g.copy()
    u = nx.Graph()
    u.add_nodes_from(d.nodes)

    protected_edges = get_protected_edges(d)
    current_undecided_edges = set(d.edges) - protected_edges

    for k in itr.count():
        print(k)
        new_undecided_edges = current_undecided_edges.copy()
        for i, j in current_undecided_edges:
            # check configuration (a)
            if set(d.predecessors(i)) - set(d.predecessors(j)) - set(d.successors(j)):
                new_undecided_edges.remove((i, j))
            # check configuration (c)
            elif set(d.successors(i)) & set(d.predecessors(j)):
                new_undecided_edges.remove((i, j))
            # check configuration (d)
            elif len(set(d.predecessors(j)) & set(u.neighbors(i))) == 2:
                new_undecided_edges.remove((i, j))
        for i, j in new_undecided_edges:
            u.add_edge(i, j)
            if d.has_edge(i, j):
                d.remove_edge(i, j)
        if current_undecided_edges == new_undecided_edges:
            break
        current_undecided_edges = new_undecided_edges

    return d, u


def get_iessgraph(g, intervention):
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

    d, u = get_essgraph(g)
    print(list(d.edges))
    print(list(u.edges))
    print(set(d.edges) | set(u.edges) == set(g.edges))




