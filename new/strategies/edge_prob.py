import random
from collections import defaultdict
import operator as op
import numpy as np
import os
from utils import graph_utils
import pandas as pd
import causaldag as cd
from typing import Dict, Any
import config

DATA_PATH = os.path.join(config.DATA_FOLDER, 'samples.csv')
INTERVENTION_PATH = os.path.join(config.DATA_FOLDER, 'interventions.csv')

def _write_data(data):
    # clear current data
    open(DATA_PATH, 'w').close()
    open(INTERVENTION_PATH, 'w').close()

    iv_nodes = []
    for iv_node, samples in data.items():
        with open(DATA_PATH, 'ab') as f:
            np.savetxt(f, samples)
        iv_nodes.extend([iv_node+1 if iv_node != -1 else -1]*len(samples))
    pd.Series(iv_nodes).to_csv(INTERVENTION_PATH, index=False)


def _load_dags(nsamples):
    dags = []
    for i in range(nsamples):
        amat = np.loadtxt('../data/TEMP_DAGS/%d.csv' % i)
        dag = graph_utils.dag_from_amat(amat)
        dags.append(dag)
    return dags


def probability_shrinkage(prob):
    return 2 * min(1 - prob, prob)


def create_edge_prob_strategy(target, n_boot):
    def edge_prob_strategy(g, data: Dict[Any, np.array], config, batch_num):
        _write_data(data)
        graph_utils.run_gies_boot(n_boot, DATA_PATH, INTERVENTION_PATH, delete=True)
        adj_mats = graph_utils.load_adj_mats()
        dags = [cd.from_amat(adj) for adj in adj_mats]
        n = config.n_samples / (config.n_batches * config.max_interventions)
        if int(n) != n:
            raise ValueError('n_samples / (n_batches * max interventions) must be an integer')
        interventions = {iv: int(n) for iv in random.sample(g.nodes, config.max_interventions)}
        parent_counts = defaultdict(int)
        node_set = dags[0].nodes
        for node in node_set:
            parent_counts[node] = 0
        for dag in dags:
            for p in dag.parents[target]:
                parent_counts[p] += 1
        parent_probs = {p: c/len(dags) for p, c in parent_counts.items()}
        print(parent_probs)
        parent_shrinkage_scores = {p: probability_shrinkage(prob) for p, prob in parent_probs.items()}
        interventions = {}
        for k in range(config.max_interventions):
            max_score = max(parent_shrinkage_scores.items(), key=op.itemgetter(1))[1]
            tied_best_ivs = [iv for iv, score in parent_shrinkage_scores.items() if score == max_score]
            best_iv = random.choice(tied_best_ivs)
            interventions[best_iv] = int(n)
            parent_shrinkage_scores.pop(best_iv)
        return interventions
    return edge_prob_strategy




