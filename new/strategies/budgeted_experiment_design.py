import operator as op
import itertools as itr
import random


def compute_bed_score(dags, cpdags, intervened_nodes):
    icpdags = [dag.interventional_cpdag(intervened_nodes, cpdag=cpdag) for dag, cpdag in zip(dags, cpdags)]
    num_oriented = [len(icpdag.arcs) - len(cpdag.arcs) for icpdag, cpdag in zip(icpdags, cpdags)]
    return sum(num_oriented)


def create_bed_strategy(dag_collection):
    cpdags = [dag.cpdag() for dag in dag_collection]

    def bed_strategy(iteration_data):
        nsamples = iteration_data.n_samples / iteration_data.n_batches
        if int(nsamples) != nsamples:
            raise ValueError('n_samples / n_batches must be an integer')
        nsamples = int(nsamples)

        remaining_intervention_ixs = set(range(len(iteration_data.intervention_set)))
        intervened_nodes = set()
        selected_intervention_ixs = set()
        for i in range(iteration_data.max_interventions):
            # print(remaining_intervention_ixs)
            scores = {
                iv_ix: compute_bed_score(dag_collection, cpdags, intervened_nodes | {iteration_data.intervention_set[iv_ix]})
                for iv_ix in remaining_intervention_ixs
            }
            sorted_scores = sorted(scores.items(), key=op.itemgetter(1), reverse=True)
            best_score = sorted_scores[0][1]
            best_nodes = list(itr.takewhile(lambda node_score: node_score[1] == best_score, sorted_scores))
            selected_iv_ix = random.choice(best_nodes)[0]
            intervened_nodes.add(iteration_data.intervention_set[selected_iv_ix])
            selected_intervention_ixs.add(iteration_data.intervention_set[selected_iv_ix])
            remaining_intervention_ixs.remove(selected_iv_ix)

        selected_interventions_to_samples = {
            iv_ix: int(nsamples/iteration_data.max_interventions)
            for iv_ix in selected_intervention_ixs
        }
        print(selected_interventions_to_samples)
        return selected_interventions_to_samples

    return bed_strategy
