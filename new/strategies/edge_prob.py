import random
import operator as op
from utils import graph_utils
import os
import numpy as np
import shutil


def create_edge_prob_strategy(target, n_boot):
    def edge_prob_strategy(iteration_data):
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




