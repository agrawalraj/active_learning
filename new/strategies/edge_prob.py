import random
from collections import defaultdict
import operator as op
from utils import graph_utils
import config
import os
import numpy as np


def create_edge_prob_strategy(target, n_boot):
    def edge_prob_strategy(iteration_data):
        # === CALCULATE NUMBER OF SAMPLES IN EACH INTERVENTION
        n = iteration_data.n_samples / (iteration_data.n_batches * iteration_data.max_interventions)
        if int(n) != n:
            raise ValueError('n_samples / (n_batches * max interventions) must be an integer')

        # === SAVE DATA, THEN CALL R CODE WITH DATA TO GET DAG SAMPLES
        print('intervened nodes:', iteration_data.current_data.keys())
        graph_utils._write_data(iteration_data.current_data)
        graph_utils.run_gies_boot(n_boot, config.TEMP_SAMPLES_PATH, config.TEMP_INTERVENTIONS_PATH, delete=True)
        dags = graph_utils._load_dags()
        dag_target_parents = [dag.parents[target] for dag in dags]
        if len(dags) != n_boot:
            raise RuntimeError('Correct number of DAGs not saved, check R code')

        # === SAVE SAMPLED DAGS FROM R FOR FUTURE REFERENCE
        for d, dag in enumerate(dags):
            amat = dag.to_amat()
            filename = os.path.join(iteration_data.strategy_folder, 'batch%d' % iteration_data.batch_num,
                                    'dag%d.npy' % d)
            np.save(filename, amat)

        parent_counts = defaultdict(int)
        for dag, target_parents in zip(dags, dag_target_parents):
            for p in target_parents:
                parent_counts[p] += 1
        parent_probs = {p: c/len(dags) for p, c in parent_counts.items()}
        print(parent_probs)
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




