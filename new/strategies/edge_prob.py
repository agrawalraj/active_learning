import random
import operator as op
from utils import graph_utils
from strategies.collect_dags import collect_dags


def create_edge_prob_strategy(target, n_boot):
    def edge_prob_strategy(iteration_data):
        # === CALCULATE NUMBER OF SAMPLES IN EACH INTERVENTION
        n = iteration_data.n_samples / (iteration_data.n_batches * iteration_data.max_interventions)
        if int(n) != n:
            raise ValueError('n_samples / (n_batches * max interventions) must be an integer')

        sampled_dags = collect_dags(iteration_data.batch_folder, iteration_data.current_data, n_boot)
        dag_target_parents = [dag.parents[target] for dag in sampled_dags]

        parent_counts = {node: 0 for node in sampled_dags[0].nodes}
        for dag, target_parents in zip(sampled_dags, dag_target_parents):
            for p in target_parents:
                parent_counts[p] += 1
        parent_probs = {p: c/len(sampled_dags) for p, c in parent_counts.items()}
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




