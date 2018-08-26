import os
from config import DATA_FOLDER
import causaldag as cd
import numpy as np
from utils import graph_utils
from analysis.rates_helper import RatesHelper


def get_dag_folders(dataset_folder):
    dags_folder = os.path.join(DATA_FOLDER, dataset_folder, 'dags')
    return [os.path.join(dags_folder, d) for d in os.listdir(dags_folder)]


def get_true_dags(dag_folders):
    true_dags = []
    for dag_folder in dag_folders:
        amat = np.loadtxt(os.path.join(dag_folder, 'adjacency.txt'))
        true_dag = cd.GaussDAG.from_amat(amat)
        true_dags.append(true_dag)
    return true_dags


def get_final_dags(strategy_folder):
    amats = [
        np.load(os.path.join(strategy_folder, 'final_dags', f))
        for f in os.listdir(os.path.join(strategy_folder, 'final_dags'))
    ]
    return [cd.GaussDAG.from_amat(amat) for amat in amats]


def get_parent_probs_by_dag(dag_folders, target, verbose=True):
    # === RECORD RESULTS OF EACH STRATEGY FOR EACH DAG
    results_by_dag = []

    for i, dag_folder in enumerate(dag_folders):
        if verbose and i % 10 == 0: print('Loading parent probabilities for DAG %d' % i)
        result_by_strategy = {}
        for filename in os.listdir(dag_folder):
            filename = os.path.join(dag_folder, filename)
            if os.path.isdir(filename):
                dags = get_final_dags(filename)
                dag_target_parents = [dag.parents[target] for dag in dags]

                # CALCULATE PARENT FREQUENCIES FROM DAGS
                parent_counts = {node: 0 for node in dags[0].nodes}
                for dag, target_parents in zip(dags, dag_target_parents):
                    for p in target_parents:
                        parent_counts[p] += 1
                parent_probs = {p: c / len(dags) for p, c in parent_counts.items()}

                result_by_strategy[os.path.basename(filename)] = parent_probs
        results_by_dag.append(result_by_strategy)

    return results_by_dag


def get_rates_by_dag(parent_probs_by_dag, true_dags, target, p_thresh=.5, verbose=True):
    result_by_dag = []
    for i, (parent_probs_by_strategy, true_dag) in enumerate(zip(parent_probs_by_dag, true_dags)):
        if verbose and i % 10 == 0: print('Loading parent probabilities for DAG %d' % i)
        true_parents = true_dag.parents[target]

        result_by_strategy = {}
        for strategy, parent_probs in parent_probs_by_strategy.items():
            positives = {p for p, prob in parent_probs.items() if prob >= p_thresh}
            negatives = {p for p, prob in parent_probs.items() if prob < p_thresh}

            true_positives = positives & true_parents
            true_negatives = negatives & (true_dag.nodes - true_parents)
            false_positives = positives & (true_dag.nodes - true_parents)
            false_negatives = negatives & true_parents

            result_by_strategy[strategy] = RatesHelper(
                true_positives=true_positives,
                true_negatives=true_negatives,
                false_positives=false_positives,
                false_negatives=false_negatives
            ).to_dict()
        result_by_dag.append(result_by_strategy)

    return result_by_dag


if __name__ == '__main__':
    dag_folders = get_dag_folders('test')
    true_dags = get_true_dags(dag_folders)
    parent_probs_by_dag = get_parent_probs_by_dag(dag_folders, 3)
    rates_by_dag = get_rates_by_dag(parent_probs_by_dag, true_dags, 3)
