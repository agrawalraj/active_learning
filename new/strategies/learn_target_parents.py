from utils import intervention_scores as scores
import numpy as np
import os
from utils import graph_utils
import pandas as pd
import causaldag as cd
import config
import operator as op
import random
from logger import LOGGER


DATA_PATH = os.path.join(config.DATA_FOLDER, 'samples.csv')
INTERVENTION_PATH = os.path.join(config.DATA_FOLDER, 'interventions.csv')


def _write_data(data):
    """
    Helper function to write the
    """
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


def create_learn_target_parents(target, n_iter=25000):
    def learn_target_parents(iteration_data):
        # === CALCULATE NUMBER OF SAMPLES IN EACH INTERVENTION
        samples_per_iv = iteration_data.n_samples / (iteration_data.n_batches * iteration_data.max_interventions)
        if int(samples_per_iv) != samples_per_iv:
            raise ValueError(
                'number of samples divided by (number of batches * max number of interventions) is not an integer')

        # === SAVE DATA, THEN CALL R CODE WITH DATA TO GET DAG SAMPLES
        _write_data(iteration_data.current_data)
        graph_utils.run_min_imap(DATA_PATH, INTERVENTION_PATH, n_iter=n_iter, delete=True)
        dags = _load_dags()
        scorer = scores.get_orient_parents_scorer(target, dags)

        # === GREEDILY SELECT INTERVENTIONS
        interventions = {}
        for k in range(iteration_data.max_interventions):
            intervention_scores = {}
            for iv in iteration_data.intervention_set:
                if iv in interventions or iv == target:
                    pass
                else:
                    intervention_scores[iv] = scorer(iv)
            LOGGER.info('intervention scores: %s' % intervention_scores)
            max_score = max(intervention_scores.items(), key=op.itemgetter(1))[1]
            tied_best_ivs = [iv for iv, score in intervention_scores.items() if score == max_score]
            best_iv = random.choice(tied_best_ivs)
            interventions[best_iv] = int(samples_per_iv)
        return interventions

    return learn_target_parents




