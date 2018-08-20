import os
import config
import causaldag as cd
import numpy as np
from utils import graph_utils
from collections import defaultdict
from analysis.rates_helper import RatesHelper


def check_gies(dag_folder, strategy, target):
    amat = np.loadtxt(os.path.join(dag_folder, 'adjacency.txt'))
    true_dag = cd.GaussDAG.from_amat(amat)
    true_parents = true_dag.parents[target]

    # === READ SAMPLES
    strategy_folder = os.path.join(dag_folder, strategy)
    samples_folder = os.path.join(strategy_folder, 'samples')
    samples = {}
    for iv in true_dag.nodes | {-1}:
        intervention_data_filename = os.path.join(samples_folder, 'intervention=%d.csv' % iv)
        if os.path.getsize(intervention_data_filename) > 0:
            samples[iv] = np.loadtxt(intervention_data_filename)
        else:
            samples[iv] = np.zeros([0, len(true_dag.nodes)])

    # === SAVE SAMPLES, THEN CALL R CODE WITH DATA TO GET DAG SAMPLES
    graph_utils._write_data(samples)
    graph_utils.run_gies_boot(10, config.TEMP_SAMPLES_PATH, config.TEMP_INTERVENTIONS_PATH, delete=True)
    amats, dags = graph_utils._load_dags()
    dag_target_parents = [dag.parents[target] for dag in dags]
    if len(dags) != 10:
        raise RuntimeError('Correct number of DAGs not saved, check R code')
    print(len(dags))

    # === CHECK PARENT PROBABILITIES
    parent_counts = {node: 0 for node in dags[0].nodes}
    for dag, target_parents in zip(dags, dag_target_parents):
        for p in target_parents:
            parent_counts[p] += 1
    parent_probs = {p: c / len(dags) for p, c in parent_counts.items()}

    positives = {p for p, prob in parent_probs.items() if prob > .5}
    negatives = {p for p, prob in parent_probs.items() if prob < .5}
    true_positives = positives & true_parents
    true_negatives = negatives & (true_dag.nodes - true_parents)
    false_positives = positives & (true_dag.nodes - true_parents)
    false_negatives = negatives & true_parents

    return parent_probs, RatesHelper(
        true_positives=true_positives,
        true_negatives=true_negatives,
        false_positives=false_positives,
        false_negatives=false_negatives
    )


if __name__ == '__main__':
    parent_probs, rh = check_gies(os.path.join(config.DATA_FOLDER, 'medium', 'dag0'), 'edge-prob', 7)

