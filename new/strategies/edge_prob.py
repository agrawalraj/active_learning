import random
from collections import defaultdict
import operator as op
import numpy as np
import os
from utils import graph_utils
import pandas as pd
import causaldag as cd
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


def _load_dags():
    """
    Helper function to load the DAGs generated in R
    """
    adj_mats = []
    paths = os.listdir(config.TEMP_DAG_FOLDER)
    for file_path in paths:
        if 'score' not in file_path and '.DS_Store' not in file_path:
            adj_mat = pd.read_csv(os.path.join(config.TEMP_DAG_FOLDER, file_path))
            adj_mats.append(adj_mat.as_matrix())
    return [cd.from_amat(adj) for adj in adj_mats]


def probability_shrinkage(prob):
    return 2 * min(1 - prob, prob)


def create_edge_prob_strategy(target, n_boot):
    def edge_prob_strategy(iteration_data):
        # === CALCULATE NUMBER OF SAMPLES IN EACH INTERVENTION
        n = iteration_data.n_samples / (iteration_data.n_batches * iteration_data.max_interventions)
        if int(n) != n:
            raise ValueError('n_samples / (n_batches * max interventions) must be an integer')

        # === SAVE DATA, THEN CALL R CODE WITH DATA TO GET DAG SAMPLES
        _write_data(iteration_data.current_data)
        graph_utils.run_gies_boot(n_boot, DATA_PATH, INTERVENTION_PATH, delete=True)
        dags = _load_dags()

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

        # === GREEDILY SELECT INTERVENTIONS
        interventions = {}
        for k in range(iteration_data.max_interventions):
            max_score = max(parent_shrinkage_scores.items(), key=op.itemgetter(1))[1]
            tied_best_ivs = [iv for iv, score in parent_shrinkage_scores.items() if score == max_score]
            best_iv = random.choice(tied_best_ivs)
            interventions[best_iv] = int(n)
            parent_shrinkage_scores.pop(best_iv)
        return interventions

    return edge_prob_strategy




