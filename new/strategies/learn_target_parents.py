from utils import graph_utils
import operator as op
import random
from collections import defaultdict
import numpy as np
from strategies.collect_dags import collect_dags


def create_learn_target_parents(target, n_boot=100):
    def learn_target_parents(iteration_data):
        # === CALCULATE NUMBER OF SAMPLES IN EACH INTERVENTION
        samples_per_iv = iteration_data.n_samples / (iteration_data.n_batches * iteration_data.max_interventions)
        if int(samples_per_iv) != samples_per_iv:
            raise ValueError(
                'number of samples divided by (number of batches * max number of interventions) is not an integer')

        dags = collect_dags(iteration_data.batch_folder, iteration_data.current_data, n_boot)

        # === GET CPDAG AND PARENTS OF EACH DAG TO USE IN CALCULATING SCORE
        cpdags = [dag.cpdag() for dag in dags]
        dag_target_parents = [dag.parents[target] for dag in dags]

        # === CALCULATE SHRINKAGE SCORES OF EACH PARENT NODE
        parent_counts = defaultdict(int)
        for dag, target_parents in zip(dags, dag_target_parents):
            for p in target_parents:
                parent_counts[p] += 1
        parent_probs = {p: c / len(dags) for p, c in parent_counts.items()}
        parent_shrinkage_scores = {p: graph_utils.probability_shrinkage(prob) for p, prob in parent_probs.items()}

        # === GREEDILY SELECT INTERVENTIONS
        selected_interventions = {}
        curr_cpdags = cpdags.copy()
        for k in range(iteration_data.max_interventions):
            intervention_scores = {}

            # === CALCULATE SCORE OF EACH INTERVENTION
            for iv in iteration_data.intervention_set:
                if iv in selected_interventions or iv == target:
                    pass
                else:
                    scores = []
                    for dag, cpdag, curr_cpdag, target_parents in zip(dags, cpdags, curr_cpdags, dag_target_parents):
                        icpdag = dag.interventional_cpdag([iv], cpdag=curr_cpdag)
                        oriented_parents = [p for p in target_parents if (p, target) in (icpdag.arcs - cpdag.arcs)]
                        score = sum(parent_shrinkage_scores[p] for p in oriented_parents)
                        scores.append(score)
                    intervention_scores[iv] = np.mean(scores)

            # === SELECT INTERVENTION WITH MAXIMUM SCORE, BREAKING TIES
            max_score = max(intervention_scores.items(), key=op.itemgetter(1))[1]
            tied_best_ivs = [iv for iv, score in intervention_scores.items() if score == max_score]
            best_iv = random.choice(tied_best_ivs)
            selected_interventions[best_iv] = int(samples_per_iv)

            if all(score == 0 for score in scores):
                print('!!! NO INTERVENTION ORIENTS ANY EDGES')
            else:
                print('SELECTED INTERVENTION: %s' % best_iv)

        return selected_interventions

    return learn_target_parents




