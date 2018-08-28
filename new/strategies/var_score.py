import numpy as np
import causaldag as cd
import operator as op
import random
import config
import os

from utils import graph_utils
from collections import defaultdict
from typing import List
from logger import LOGGER
from causaldag import BinaryIntervention, ConstantIntervention


def node_iv_var_mat(adj_mat, node_vars, iv_strengths, n_monte_carlo=1000):
    p = adj_mat.shape[0]
    gdag = cd.GaussDAG.from_amat(adj_mat)
    var_mat = np.zeros((p, p))
    ivs = []
    for i, iv_strength in enumerate(iv_strengths):
        ivs.append(
            BinaryIntervention(
                ConstantIntervention(val=iv_strength).sample,
                ConstantIntervention(val=-iv_strength).sample
            )
        )

    for ix, iv in enumerate(ivs):
        iv_samps = gdag.sample_interventional({ix: iv.sample}, n_monte_carlo)
        var_mat[:, ix] = np.multiply(np.var(iv_samps, axis=0), 1 / node_vars)

    return var_mat


def get_orient_mask(target, adj_mat):
    mask = np.zeros(adj_mat.shape)
    parents_mask = adj_mat[:, target]
    parents_mask = parents_mask != 0
    dag = cd.DAG.from_amat(adj_mat)
    cpdag = dag.cpdag() 
    for iv in range(adj_mat.shape[0]):
        if iv != target:
            icpdag = dag.interventional_cpdag([iv], cpdag=cpdag)
            oriented_parents = [p for p in dag.parents[target] if (p, target) in (icpdag.arcs - cpdag.arcs)]
            for parent, parent_mask in enumerate(parents_mask):
                if parent_mask != False and parent in oriented_parents:
                    mask[parent, iv] = 1
    return mask


def var_score_mat(target, adj_mats, node_vars, iv_strengths):
    node_vars = np.array(node_vars) # Make sure right type
    iv_strengths = np.array(iv_strengths)
    p = adj_mats[0].shape[0]
    iv_scores = np.zeros((p, p))
    num_adj_mats = len(adj_mats)
    for adj_mat in adj_mats:
        parent_orient_mask = get_orient_mask(target, adj_mat)
        var_mat = node_iv_var_mat(adj_mat, node_vars, iv_strengths)
        orient_var_mat = np.multiply(var_mat, parent_orient_mask)
        iv_scores += orient_var_mat / num_adj_mats
    return iv_scores


def create_var_score_fn(parent_shrinkage_scores, target, adj_mats, node_vars, iv_strengths):
    p = adj_mats[0].shape[0]
    iv_scores = var_score_mat(target, adj_mats, node_vars, iv_strengths)
    for node in range(p): # Don't include target node
        if node != target:
            iv_scores[node, :] = iv_scores[node, :] * parent_shrinkage_scores[node]

    def var_score_fn(interventions):
        return np.sum(np.max(iv_scores[:, interventions], axis=1))

    return var_score_fn


def greedy_iv(int_score_fn, iv_family, K):
    interventions = set()
    while len(interventions) < K: 
        iv_scores = {}
        for iv in iv_family.difference(interventions):
            prev_plus_cur_iv = interventions.copy()
            prev_plus_cur_iv.add(iv)
            iv_scores[iv] = int_score_fn(list(prev_plus_cur_iv))
        max_score = max(iv_scores.items(), key=op.itemgetter(1))[1]
        tied_best_ivs = [iv for iv, score in iv_scores.items() if score == max_score]
        best_iv = random.choice(tied_best_ivs)
        interventions.add(best_iv)
    return list(interventions)


def create_variance_strategy(target, node_vars, iv_strengths, n_boot=100):
    def variance_strategy(iteration_data):
        # === CALCULATE NUMBER OF SAMPLES IN EACH INTERVENTION
        n = iteration_data.n_samples / (iteration_data.n_batches * iteration_data.max_interventions)
        if int(n) != n:
            raise ValueError('n_samples / (n_batches * max interventions) must be an integer')

        # === DEFINE PATHS FOR FILES WHICH WILL HOLD THE TEMPORARY DATA
        samples_path = os.path.join(iteration_data.batch_folder, 'samples.csv')
        interventions_path = os.path.join(iteration_data.batch_folder, 'interventions.csv')
        dags_path = os.path.join(iteration_data.batch_folder, 'TEMP_DAGS/')

        # === SAVE DATA, THEN CALL R CODE WITH DATA TO GET DAG SAMPLES
        graph_utils._write_data(iteration_data.current_data, samples_path, interventions_path)
        graph_utils.run_gies_boot(n_boot, samples_path, interventions_path, dags_path, delete=True)
        amats, dags = graph_utils._load_dags(dags_path, delete=True)
        dag_target_parents = [dag.parents[target] for dag in dags]
        if len(dags) != n_boot:
            raise RuntimeError('Correct number of DAGs not saved, check R code')

        # === SAVE SAMPLED DAGS FROM R FOR FUTURE REFERENCE
        for d, amat in enumerate(amats):
            np.save(os.path.join(iteration_data.batch_folder, 'dag%d.npy' % d), amat)

        parent_counts = {node: 0 for node in dags[0].nodes}
        for dag, target_parents in zip(dags, dag_target_parents):
            for p in target_parents:
                parent_counts[p] += 1
        parent_probs = {p: c/len(dags) for p, c in parent_counts.items()}
        parent_shrinkage_scores = {p: graph_utils.probability_shrinkage(prob) for p, prob in parent_probs.items()}
        var_score_fn = create_var_score_fn(parent_shrinkage_scores, target, amats, node_vars, iv_strengths)
        p = amats[0].shape[0]
        iv_family = set()
        [iv_family.add(iv) for iv in range(p) if iv != target]
        interventions = greedy_iv(var_score_fn, iv_family, iteration_data.max_interventions)
        selected_interventions = {}
        for iv in interventions:
            selected_interventions[iv] = int(n)
        return selected_interventions
    return variance_strategy


if __name__ == '__main__':
    B = np.zeros((3, 3))
    B[0, 2] = 1
    B[1, 2] = 1
    B[0, 1] = 2
    gdag = cd.GaussDAG.from_amat(B)
    obs_samps = gdag.sample(10000)
    node_vars = np.var(obs_samps, axis=0) # exact is (1, 5, 11)
    iv_strengths = [2, 3, 1]


