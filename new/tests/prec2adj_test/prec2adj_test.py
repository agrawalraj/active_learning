import unittest
from utils import graph_utils
import numpy as np
import random
import ipdb


def switch_perm(curr_perm, i, j):
    a, b = curr_perm.index(i), curr_perm.index(j)
    new_perm = curr_perm.copy()
    new_perm[a], new_perm[b] = j, i
    return new_perm


class TestPrec2Adj(unittest.TestCase):
    # def test_stays_same(self):
    #     p = 10
    #     edge_prob = .5
    #     ndags = 100
    #     for _ in range(ndags):
    #         g = graph_utils.random_graph(p, edge_prob)
    #         adj = graph_utils.random_adj(g)
    #         prec = graph_utils.adj2prec(adj)
    #         adj2, _ = graph_utils.prec2adj(prec, range(10))
    #         self.assertTrue(np.allclose(adj, adj2))
    #
    # def test_same_siginv(self):
    #     p = 10
    #     edge_prob = .5
    #     ndags = 100
    #     for _ in range(ndags):
    #         g = graph_utils.random_graph(p, edge_prob)
    #         adj = graph_utils.random_adj(g)
    #         prec = graph_utils.adj2prec(adj)
    #         cov_edges = graph_utils.get_covered_edges(g)
    #         if len(cov_edges) == 0:
    #             continue
    #         i, j = random.sample(list(cov_edges), 1)[0]
    #         adj2, omega2 = graph_utils.prec2adj(prec, switch_perm(list(range(10)), i, j))
    #         siginv1 = graph_utils.adj2prec(adj)
    #         siginv2 = graph_utils.adj2prec(adj2, omega2)
    #         self.assertTrue(np.allclose(siginv1, siginv1))

    def test_same_num_edges(self):
        p = 5
        edge_prob = .5
        ndags = 100
        for _ in range(ndags):
            g = graph_utils.random_graph(p, edge_prob)
            adj = graph_utils.random_adj(g)
            prec = graph_utils.adj2prec(adj)
            cov_edges = graph_utils.get_covered_edges(g)
            if len(cov_edges) == 0:
                continue
            i, j = random.sample(list(cov_edges), 1)[0]
            adj2, omega2 = graph_utils.prec2adj(prec, switch_perm(list(range(p)), i, j))
            adj2[abs(adj2) < 1e-10] = 0
            n_edges1 = adj.astype(bool).sum()
            n_edges2 = adj2.astype(bool).sum()
            if n_edges1 != n_edges2:
                print('cov edge:', i, j)
                print(n_edges1, n_edges2)
                print(adj.astype(bool).astype(int))
                print(adj2.astype(bool).astype(int))
                ipdb.set_trace()
            self.assertTrue(n_edges1 == n_edges2)

    # def test_one_diff_edge(self):
    #     p = 10
    #     edge_prob = .5
    #     ndags = 100
    #     for _ in range(ndags):
    #         g = graph_utils.random_graph(p, edge_prob)
    #         adj = graph_utils.random_adj(g)
    #         prec = graph_utils.adj2prec(adj)
    #         cov_edges = graph_utils.get_covered_edges(g)
    #         if len(cov_edges) == 0:
    #             continue
    #         i, j = random.sample(list(cov_edges), 1)[0]
    #         adj2, omega2 = graph_utils.prec2adj(prec, switch_perm(list(range(10)), i, j))
    #         adj2[abs(adj2) < 1e-10] = 0
    #         diff_edges = (adj2.astype(bool) != adj.astype(bool)).sum()
    #         print(diff_edges)
    #         if diff_edges != 2:
    #             ipdb.set_trace()
    #         self.assertTrue(diff_edges == 2)


if __name__ == '__main__':
    unittest.main()





