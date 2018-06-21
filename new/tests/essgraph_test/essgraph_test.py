from utils import graph_utils as gu
import unittest
import numpy as np
import os
import itertools as itr


class TestEssgraph(unittest.TestCase):
    def test_essgraph(self):
        p = 10
        edge_probs = [.5]
        n_dags = 10
        for edge_prob in edge_probs:
            for i in range(n_dags):
                print('==========')
                g = gu.random_graph(p, edge_prob)
                adj = gu.graph2adj(g)
                np.savetxt('tests/essgraph_test/tmp-graph.txt', adj)
                os.system('R -f tests/essgraph_test/get_essgraph.R > /dev/null')
                r_essgraph = np.loadtxt('tests/essgraph_test/tmp-graph-r.txt')
                r_directed_edges = []
                r_undirected_edges = []
                for i, j in itr.combinations(range(p), 2):
                    if r_essgraph[i, j] == 1:
                        if r_essgraph[j, i] == 1:
                            r_undirected_edges.append((i, j))
                        else:
                            r_directed_edges.append((i, j))
                print(r_undirected_edges)
                print(len(r_undirected_edges))

                d, u = gu.get_essgraph(g, verbose=True)
                directed_edges = d.edges
                undirected_edges = u.edges
                print(undirected_edges)
                print(len(undirected_edges))

        os.system('rm tests/essgraph_test/tmp-graph.txt')
        os.system('rm tests/essgraph_test/tmp-graph-r.txt')


if __name__ == '__main__':
    unittest.main()

