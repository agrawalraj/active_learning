from utils import intervention_scores as scores
from utils import graph_utils
import config
import operator as op
import random
from logger import LOGGER


def create_learn_target_parents(target, n_iter=25000):

    def learn_target_parents(iteration_data):
        # === CALCULATE NUMBER OF SAMPLES IN EACH INTERVENTION
        samples_per_iv = iteration_data.n_samples / (iteration_data.n_batches * iteration_data.max_interventions)
        if int(samples_per_iv) != samples_per_iv:
            raise ValueError(
                'number of samples divided by (number of batches * max number of interventions) is not an integer')

        # === SAVE DATA, THEN CALL R CODE WITH DATA TO GET DAG SAMPLES
        graph_utils._write_data(iteration_data.current_data)
        graph_utils.run_min_imap(config.TEMP_SAMPLES_PATH, config.TEMP_INTERVENTIONS_PATH, n_iter=n_iter, delete=True)
        dags = graph_utils._load_dags()
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




