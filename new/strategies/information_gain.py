from strategies.collect_dags import collect_dags
from utils import graph_utils
import xarray as xr
import numpy as np
import itertools as itr
from collections import defaultdict
from scipy.misc import logsumexp
from scipy import special
import random


def binary_entropy(probs):
    return special.entr(probs) - special.xlog1py(1 - probs, -probs)


def create_info_gain_strategy(precision_matrix, n_boot, graph_functionals):
    def info_gain_strategy(iteration_data):
        # === CALCULATE NUMBER OF SAMPLES IN EACH INTERVENTION
        nsamples = iteration_data.n_samples / (iteration_data.n_batches * iteration_data.max_interventions)
        if int(nsamples) != nsamples:
            raise ValueError('n_samples / (n_batches * max interventions) must be an integer')

        sampled_dags = collect_dags(iteration_data.batch_folder, iteration_data.current_data, n_boot)
        gauss_dags = [graph_utils.prec2dag(precision_matrix, dag.topological_sort()) for dag in sampled_dags]

        # == CREATE MATRIX MAPPING EACH GRAPH TO 0 or 1 FOR THE SPECIFIED FUNCTIONALS
        functional_matrix = np.zeros([n_boot, len(graph_functionals)])
        for (dag_ix, dag), (functional_ix, functional) in itr.product(enumerate(gauss_dags), enumerate(graph_functionals)):
            functional_matrix[dag_ix, functional_ix] = functional(dag)

        # === FOR EACH GRAPH, OBTAIN SAMPLES FOR EACH INTERVENTION THAT'LL BE USED TO BUILD UP THE HYPOTHETICAL DATASET
        datapoints = [
            [
                dag.sample_interventional(intervention, nsamples=nsamples)
                for intervention in iteration_data.interventions
            ]
            for dag in gauss_dags
        ]
        logpdfs = xr.DataArray(
            np.zeros([n_boot, len(iteration_data.intervention_set), n_boot, nsamples]),
            dims=['outer_dag', 'intervened_node', 'inner_dag', 'datapoint'],
            coords={
                'outer_dag': list(range(n_boot)),
                'intervention': list(range(iteration_data.interventions)),
                'inner_dag': list(range(n_boot)),
                'datapoint': list(range(nsamples))
            }
        )
        for outer_dag_ix in range(n_boot):
            for intervention_ix, intervention in enumerate(iteration_data.interventions):
                for inner_dag_ix, inner_dag in enumerate(gauss_dags):
                    loc = dict(outer_dag=outer_dag_ix, intervention=intervention_ix, inner_dag=inner_dag_ix)
                    logpdfs.loc[loc] = inner_dag.logpdf(datapoints[outer_dag_ix][intervention_ix], interventions=intervention)

        current_logpdfs = np.zeros([n_boot, n_boot])
        selected_interventions = defaultdict(int)
        for sample_num in range(nsamples):
            intervention_scores = np.zeros(len(iteration_data.interventions))
            intervention_logpdfs = np.zeros([len(iteration_data.interventions), n_boot, n_boot])
            for intv_ix in range(len(iteration_data.interventions)):
                for outer_dag_ix in range(n_boot):
                    # current number of times this intervention has already been selected
                    datapoint_ix = selected_interventions[intv_ix]

                    intervention_logpdfs[intv_ix][outer_dag_ix] = logpdfs.sel(
                        outer_dag=outer_dag_ix,
                        intervention=intv_ix,
                        datapoint=datapoint_ix
                    )
                    new_logpdfs = current_logpdfs[outer_dag_ix] + intervention_logpdfs[intv_ix][outer_dag_ix]

                    importance_weights = np.exp(new_logpdfs - logsumexp(new_logpdfs))
                    functional_probabilities = (importance_weights[:, np.newaxis] * functional_matrix).sum(axis=0)
                    functional_entropies = binary_entropy(functional_probabilities)
                    intervention_scores[intv_ix] += functional_entropies.sum()

            max_scoring_interventions = np.nonzero(intervention_scores == intervention_scores.max())[0]
            selected_intv_ix = random.choice(max_scoring_interventions)
            current_logpdfs = current_logpdfs + intervention_logpdfs[selected_intv_ix]
            selected_interventions[selected_intv_ix] += 1

        return selected_interventions

    return info_gain_strategy
