from utils import intervention_scores as scores
import numpy as np
import os
from utils import graph_utils
import pandas as pd


def _write_data(data):
    n_nodes = len(data)
    all_samples = []
    iv_nodes = []
    for iv_node, samples in data:
        all_samples.extend(samples)
        iv_nodes.extend([iv_node]*len(samples))
    df = pd.DataFrame(all_samples, columns=[str(i) for i in range(n_nodes)])
    df.to_csv('data/tmp-data/samples.csv')
    pd.Series(iv_nodes).to_csv('data/tmp-data/interventions.csv', index=False)


def _load_dags(nsamples):
    dags = []
    for i in range(nsamples):
        amat = np.loadtxt('../data/TEMP_DAGS/%d.csv' % i)
        dag = graph_utils.dag_from_amat(amat)
        dags.append(dag)
    return dags


def learn_target_parents(g, target, data, config):
    _write_data(data)
    os.system('R')

    dags = _load_dags()
    scorer = scores.get_orient_parents_scorer(target, dags)

    samples_per_iv = config.n_samples / (config.n_batches * config.max_interventions)
    if int(samples_per_iv) != samples_per_iv:
        raise ValueError(
            'number of samples divided by (number of batches * max number of interventions) is not an integer')
    n_samples = [int(samples_per_iv) for k in range(config.max_interventions)]

    interventions = []
    for k in range(config.max_interventions):
        intervention_scores = [scorer(node) if node not in interventions else 0 for node in range(config.n_nodes)]
        best_intervention = np.argmax(intervention_scores)
        interventions.append(best_intervention)

    return interventions, n_samples




