import os
from config import DATA_FOLDER
import causaldag as cd
import numpy as np
from utils import graph_utils
from analysis.rates_helper import RatesHelper
import xarray as xr
import json


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

def get_l1_score(parent_probs, dag, target):
    score = 0
    true_parents = dag.parents[target]
    nonparents = set(dag.nodes) - set(true_parents)
    for parent in true_parents:
        score += 1 - parent_probs[parent]
    for nonparent in nonparents:
        score += parent_probs[nonparent]
    return score


def get_parent_probs_by_dag(dag_folders, target, verbose=True):
    # === RECORD RESULTS OF EACH STRATEGY FOR EACH DAG
    results_by_dag = []

    for i, dag_folder in enumerate(dag_folders):
        print(dag_folder)
        result_by_strategy = {}
        for filename in os.listdir(dag_folder):
            filename = os.path.join(dag_folder, filename)
            if os.path.isdir(filename):
                final_parent_probs_filename = os.path.join(filename, 'parent_probs.json')
                if not os.path.exists(final_parent_probs_filename):
                    dags = get_final_dags(filename)
                    dag_target_parents = [dag.parents[target] for dag in dags]

                    # CALCULATE PARENT FREQUENCIES FROM DAGS
                    parent_counts = {node: 0 for node in dags[0].nodes}
                    for dag, target_parents in zip(dags, dag_target_parents):
                        for p in target_parents:
                            parent_counts[p] += 1
                    parent_probs = {p: c / len(dags) for p, c in parent_counts.items()}
                    json.dump(parent_probs, open(final_parent_probs_filename, 'w'), indent=2)
                else:
                    parent_probs = {int(node): val for node, val in json.load(open(final_parent_probs_filename)).items()}

                result_by_strategy[os.path.basename(filename)] = parent_probs
        results_by_dag.append(result_by_strategy)

    return results_by_dag


def get_rates_data_array(parent_probs_by_dag, true_dags, target, strategy_names, ks, bs, ns, alphas, verbose=True):
    """
    Return a DataArray with dimensions:
    (strategy_names, ks, bs, ns, p_threshes, rates, trues_dags)
    with the value of each rate statistic in each position
    """
    selected_rates = ['tpr', 'fpr', 'tnr', 'fnr', 'ppv', 'npv']
    rate_array = np.zeros([
        len(strategy_names),
        len(ks),
        len(bs),
        len(ns),
        len(alphas),
        len(selected_rates),
        len(true_dags)
    ])
    s2ix = {s: ix for ix, s in enumerate(strategy_names)}
    k2ix = {k: ix for ix, k in enumerate(ks)}
    b2ix = {b: ix for ix, b in enumerate(bs)}
    n2ix = {n: ix for ix, n in enumerate(ns)}
    alpha2ix = {alpha: ix for ix, alpha in enumerate(alphas)}
    rate2ix = {r: ix for ix, r in enumerate(selected_rates)}

    # ITERATE OVER ALL DAGS
    for dag_num, (parent_probs_by_strategy, true_dag) in enumerate(zip(parent_probs_by_dag, true_dags)):
        true_parents = true_dag.parents[target]
        true_nonparents = (set(true_dag.nodes) - true_parents - {target})

        # ITERATE THROUGH EACH STRATEGY
        for strategy, parent_probs in parent_probs_by_strategy.items():
            # GET PARAMETERS USED BY STRATEGY
            print(strategy)
            strategy_name, n_str, b_str, k_str = strategy.split(',')
            k = k_str[2:]
            k = int(k) if k != 'None' else None
            b = int(b_str[2:])
            n = int(n_str[2:])

            # ITERATE OVER EACH ALPHA
            for alpha in alphas:
                labelled_positives = {p for p, prob in parent_probs.items() if prob >= alpha}
                labelled_negatives = set(parent_probs.keys()) - labelled_positives

                true_positives = labelled_positives & true_parents
                true_negatives = labelled_negatives & true_nonparents
                false_positives = labelled_positives & true_nonparents
                false_negatives = labelled_negatives & true_parents

                rh = RatesHelper(
                    true_positives=true_positives,
                    true_negatives=true_negatives,
                    false_positives=false_positives,
                    false_negatives=false_negatives
                ).to_dict()
                for rate in selected_rates:
                    s_ix = s2ix[strategy_name]
                    k_ix = k2ix[k]
                    b_ix = b2ix[b]
                    n_ix = n2ix[n]
                    a_ix = alpha2ix[alpha]
                    r_ix = rate2ix[rate]
                    rate_array[
                        s_ix,
                        k_ix,
                        b_ix,
                        n_ix,
                        a_ix,
                        r_ix,
                        dag_num
                    ] = rh[rate]

    return xr.DataArray(
        rate_array,
        coords=[strategy_names, ks, bs, ns, alphas, selected_rates, range(len(true_dags))],
        dims=['strategy', 'k', 'b', 'n', 'alpha', 'rate', 'dag']
    )


if __name__ == '__main__':
    dag_folders = get_dag_folders('test')
    true_dags = get_true_dags(dag_folders)
    parent_probs_by_dag = get_parent_probs_by_dag(dag_folders, 3)
    rates_da = get_rates_data_array(
        parent_probs_by_dag,
        true_dags,
        target=3,
        strategy_names=['random', 'edge-prob'],
        ks=[2],
        bs=[1],
        ns=[30, 60],
        alphas=np.linspace(0, 1, 11)
    )
