import numpy as np
from utils import graph_utils
from collections import defaultdict


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


def get_orient_parents_scorer(target, dags):
    parent_counts = defaultdict(int)
    for dag in dags:
        for p in dag.predecessors[target]:
            parent_counts[p] += 1
    parent_probs = {p: c/len(dags) for p, c in parent_counts.items()}
    parent_shrinkage_scores = {p: probability_shrinkage(prob) for p, prob in parent_probs.items()}

    def scorer(intervention):
        scores = []
        for dag in dags:
            essgraph_dir, essgraph_undir = graph_utils.get_essgraph(dag)
            iessgraph_dir, iessgraph_undir = graph_utils.get_iessgraph(dag, [intervention])
            parents = dag.predecessors(target)
            parents_oriented_by_intervention = [
                p for p in parents
                if not essgraph.has_edge(p, target) and iessgraph_dir.has_edge(p, target)
            ]
            score = sum(parent_shrinkage_scores[p] for p in parents_oriented_by_intervention)
            scores.append(score)

        return np.mean(scores)

    return scorer


def probability_shrinkage(prob):
    return 2 * min(1 - prob, prob)


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





