import numpy as np
from utils import graph_utils
from collections import defaultdict
import causaldag as cd
from typing import List
from logger import LOGGER


def get_orient_parents_scorer(target, dags: List[cd.DAG]):
    parent_counts = defaultdict(int)
    for dag in dags:
        for p in dag.parents[target]:
            parent_counts[p] += 1
    parent_probs = {p: c/len(dags) for p, c in parent_counts.items()}
    parent_shrinkage_scores = {p: probability_shrinkage(prob) for p, prob in parent_probs.items()}
    print(parent_probs)
    def scorer(intervention):
        scores = []
        for dag in dags:
            LOGGER.info(dag)
            if intervention == -1:  # observational
                icpdag = dag.cpdag()
            else:
                icpdag = dag.interventional_cpdag([intervention])
            parents = dag.parents[target]
            parents_oriented_by_intervention = [
                p for p in parents
                if (p, target) in icpdag.arcs
            ]
            LOGGER.info('parents oriented by intervention %d: %s' % (intervention, parents_oriented_by_intervention))
            score = sum(parent_shrinkage_scores[p] for p in parents_oriented_by_intervention)
            scores.append(score)

        return np.mean(scores) if len(scores) != 0 else 0

    return scorer


def probability_shrinkage(prob):
    return 2 * min(1 - prob, prob)
