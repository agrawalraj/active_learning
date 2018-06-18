import numpy as np
from utils import graph_utils


def bed_score(intervention, dags, essgraph):
    scores = []
    essgraph_dir, essgraph_undir = essgraph
    for dag in dags:
        iessgraph_dir, iessgraph_undir = graph_utils.get_iessgraph(dag, intervention)
        scores.append(len(iessgraph_dir.edges) - len(essgraph_dir))
    return np.mean(scores)
