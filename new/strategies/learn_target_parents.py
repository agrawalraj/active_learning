from utils import intervention_scores as scores
import numpy as np
import os
from utils import graph_utils
import pandas as pd
import causaldag as cd
from typing import Dict, Any
import config
import operator as op
import random
from logger import LOGGER


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


def create_learn_target_parents(target, n_iter=25000):
    def learn_target_parents(g, data: Dict[Any, np.array], config, batch_num):
        _write_data(data)
        graph_utils.run_min_imap(DATA_PATH, INTERVENTION_PATH, n_iter=n_iter, delete=True)
        adj_mats = graph_utils.load_adj_mats()
        dags = [cd.from_amat(adj) for adj in adj_mats]
        scorer = scores.get_orient_parents_scorer(target, dags)

        samples_per_iv = config.n_samples / (config.n_batches * config.max_interventions)
        if int(samples_per_iv) != samples_per_iv:
            raise ValueError(
                'number of samples divided by (number of batches * max number of interventions) is not an integer')

        interventions = {}
        for k in range(config.max_interventions):
            intervention_scores = {
                node: scorer(node) if node not in interventions else 0
                for node in [-1, *range(config.n_nodes)]
            }
            LOGGER.info('intervention scores: %s' % intervention_scores)
            max_score = max(intervention_scores.items(), key=op.itemgetter(1))[1]
            tied_best_ivs = [iv for iv, score in intervention_scores.items() if score == max_score]
            best_iv = random.choice(tied_best_ivs)
            interventions[best_iv] = int(samples_per_iv)

        return interventions

    return learn_target_parents




