import numpy as np
from utils import graph_utils


def bed_score(intervention, dags, essgraph):
    scores = []
    essgraph_dir, essgraph_undir = essgraph
    for dag in dags:
        iessgraph_dir, iessgraph_undir = graph_utils.get_iessgraph(dag, [intervention])
        scores.append(len(essgraph_undir.edges) - len(iessgraph_undir.edges))
    return np.mean(scores)


def bed_score_targeted(intervention, dags, essgraph, target):
    scores = []
    essgraph_dir, essgraph_undir = essgraph
    for dag in dags:
        iessgraph_dir, iessgraph_undir = graph_utils.get_iessgraph(dag, [intervention])
        parents = dag.predecessors(target)
        n_undirected_parents_old = sum(essgraph_undir.has_edge(p, target) for p in parents)
        n_undirected_parents_new = sum(iessgraph_undir.has_edge(p, target) for p in parents)
        score = n_undirected_parents_old - n_undirected_parents_new
        scores.append(score)
    return np.mean(scores)


if __name__ == '__main__':
    import networkx as nx
    dag1 = nx.DiGraph()
    dag1.add_edges_from([
        (0, 1),
        (0, 2),
        (0, 3),
        (2, 3)
    ])
    dag2 = nx.DiGraph()
    dag2.add_edges_from([
        (0, 1),
        (0, 2),
        (0, 3),
        (3, 2)
    ])
    essgraph = graph_utils.get_essgraph(dag1)
    dags = bed_score_targeted(0, [dag1, dag2], essgraph, 3)





