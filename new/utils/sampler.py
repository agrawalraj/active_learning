from __future__ import division  # in case python2 is used

from utils.graph_utils import *
import numpy as np
import networkx as nx

def sample_dags(g0, siginv, data, burn_in=100, thin_factor=20, iterations=1000):
    g_curr = g0.copy()
    cov_edges_curr = list(get_covered_edges(g_curr))
    node_order_curr = list(nx.topological_sort(g_curr))
    sample_dags = []
    probs = []
    for t in range(iterations):
        # Randomly select a covered edge to flip
        np.random.shuffle(cov_edges_curr)
        xi, xj = cov_edges_curr[0]
        g_next = g_curr.copy()
        reverse_edge(g_next, xi, xj)
        cov_edges_next = list(update_covered_edges(g_next, xi, xj, set(cov_edges_curr)))
        node_order_next = update_order(node_order_curr, xi, xj)

        # Compute the acceptance probability
        print(set(g_next.edges) - set(g_curr.edges))
        p_curr_unnormalized = compute_log_posterior_unnormalized(g_curr, node_order_curr, siginv, data)
        p_next_unnormalized = compute_log_posterior_unnormalized(g_next, node_order_next, siginv, data)
        print(p_curr_unnormalized, p_next_unnormalized)
        accept_prob = len(cov_edges_curr) / len(cov_edges_next) \
                      * np.exp(p_next_unnormalized - p_curr_unnormalized)
        accept_prob = np.min([1, accept_prob])

        # Make an MCMC step
        accepted = np.random.binomial(1, accept_prob, 1)[0]
        if accepted == 1:
            g_curr = g_next.copy()  # should we copy this graph?
            cov_edges_curr = cov_edges_next.copy()
            p_curr_unnormalized = p_next_unnormalized
            node_order_curr = node_order_next
        if t > burn_in and t % thin_factor == 0:
            sample_dags.append(g_curr)
            probs.append(p_curr_unnormalized)
    return sample_dags, probs


def sample_dags_uniform(g0):
    return sample_dags(g0, [])


if __name__ == '__main__':
    g0 = random_graph(10, .5)
    adj_mat = random_adj(g)
    omega = np.random.uniform(.5, 1, 10)
    omega = np.diag(omega)
    siginv = adj2prec(adj_mat, omega)
    data = sample_graph_int(g, adj_mat, [2, 4, 5, 9], [5]*4) 
    sample_dags(g0, siginv, data)

