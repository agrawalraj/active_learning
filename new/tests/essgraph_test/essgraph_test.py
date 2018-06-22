from utils import graph_utils as gu
import unittest
import numpy as np
import os
import itertools as itr
import networkx as nx


def get_r_essgraph(g):
    adj = gu.graph2adj(g)
    np.savetxt('tests/essgraph_test/tmp-graph.txt', adj)
    os.system('R -f tests/essgraph_test/get_essgraph.R > /dev/null')
    r_essgraph = np.loadtxt('tests/essgraph_test/tmp-graph-r.txt')
    r_directed_edges = set()
    r_undirected_edges = set()
    for i, j in itr.combinations(range(adj.shape[0]), 2):
        if r_essgraph[i, j] == 1:
            if r_essgraph[j, i] == 1:
                r_undirected_edges.add((i, j))
            else:
                r_directed_edges.add((i, j))
    return r_directed_edges, r_undirected_edges


class TestEssgraph(unittest.TestCase):
    def test_essgraph(self):
        p = 5
        edge_probs = [.5]
        n_dags = 10
        for edge_prob in edge_probs:
            for i in range(n_dags):
                print('==========')
                g = gu.random_graph(p, edge_prob)
                print(g.edges)
                r_directed_edges, r_undirected_edges = get_r_essgraph(g)

                d, u = gu.get_essgraph(g, verbose=False)
                directed_edges = set(d.edges)
                undirected_edges = set(u.edges)
                print('r_undir', r_undirected_edges)
                print('undir', undirected_edges)
                print('r_dir', r_directed_edges)
                print('dir', directed_edges)
                self.assertEqual(r_undirected_edges, undirected_edges)
                # self.assertEqual(r_directed_edges, directed_edges)

        os.system('rm tests/essgraph_test/tmp-graph.txt')
        os.system('rm tests/essgraph_test/tmp-graph-r.txt')

    def test_essgraph1(self):
        g = nx.DiGraph()
        g.add_edges_from([
            (0, 1),
            (1, 2),
            (0, 3),
            (1, 2),
            (1, 4),
            (2, 3)
        ])
        d, u = gu.get_essgraph(g)
        d = set(d.edges)
        u = set(u.edges)
        d_r, u_r = get_r_essgraph(g)
        print(u)
        print(u_r)


if __name__ == '__main__':
    unittest.main()

