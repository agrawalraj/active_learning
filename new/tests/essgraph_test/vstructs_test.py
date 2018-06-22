from utils import graph_utils as gu
import unittest
import numpy as np
import os
import itertools as itr


class TestEssgraph(unittest.TestCase):
    def test_essgraph(self):
        p = 5
        edge_probs = [.5]
        n_dags = 10
        for edge_prob in edge_probs:
            for i in range(n_dags):
                g = gu.random_graph(p, edge_prob)
                adj = gu.graph2adj(g)
                np.savetxt('tests/essgraph_test/tmp-graph.txt', adj)
                os.system('R -f tests/essgraph_test/get_vstructs.R > /dev/null')
                r_vstructs = np.loadtxt('tests/essgraph_test/vstructs-r.txt', dtype=str)
                r_vstructs = {(int(i[1:-1])-1, int(j[1:-1])-1) for i, j in r_vstructs}

                v_structs = gu.get_vstructures(g)
                self.assertEqual(r_vstructs, v_structs)

        os.system('rm tests/essgraph_test/tmp-graph.txt')
        os.system('rm tests/essgraph_test/vstructs-r.txt')


if __name__ == '__main__':
    unittest.main()
