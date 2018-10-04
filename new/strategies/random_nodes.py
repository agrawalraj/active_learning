import random
from collections import Counter
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class IterationData:
    current_data: Dict[Any, np.array]
    max_interventions: int
    n_samples: int
    batch_num: int
    n_batches: int
    intervention_set: list
    interventions: list
    batch_folder: str
    precision_matrix: np.ndarray


def random_strategy(iteration_data):
    if iteration_data.max_interventions is None:
        n = iteration_data.n_samples / iteration_data.n_batches
        if int(n) != n:
            raise ValueError('n_samples / n_batches must be an integer')
        intv_ixs = list(range(len(iteration_data.intervention_set)))
        return dict(Counter(random.choices(intv_ixs, k=int(n))))
    else:
        nsamples = iteration_data.n_samples / iteration_data.n_batches
        if int(nsamples) != nsamples:
            raise ValueError('n_samples / n_batches must be an integer')

        n_ivs = len(iteration_data.intervention_set)
        k = iteration_data.max_interventions if n_ivs > iteration_data.max_interventions else n_ivs
        ns = [int(np.ceil(nsamples/k)) for _ in range(k)]
        if sum(ns) != nsamples:
            ns[-1] -= sum(ns) - nsamples

        intv_ixs = list(range(len(iteration_data.intervention_set)))
        interventions = {
            iteration_data.intervention_set[intv_ix]: int(n)
            for intv_ix, n in zip(random.sample(intv_ixs, k), ns)
        }

        return interventions


def _non_isolated_nodes(cpdag):
    return {node for node in cpdag.nodes if cpdag.undirected_neighbors[node]}


def create_random_smart_strategy(cpdag):
    def random_smart_strategy(iteration_data):
        non_isolated_nodes = _non_isolated_nodes(cpdag)
        modified_iv_ixs = [ix for ix, node in enumerate(iteration_data.intervention_set) if node in non_isolated_nodes]
        modified_intervention_set = [iteration_data.intervention_set[ix] for ix in modified_iv_ixs]
        modified_interventions = [iteration_data.interventions[ix] for ix in modified_iv_ixs]

        iteration_data_new = IterationData(
            current_data=iteration_data.current_data,
            max_interventions=iteration_data.max_interventions,
            n_samples=iteration_data.n_samples,
            batch_num=iteration_data.batch_num,
            n_batches=iteration_data.n_batches,
            intervention_set=modified_intervention_set,
            interventions=modified_interventions,
            batch_folder=iteration_data.batch_folder,
            precision_matrix=iteration_data.precision_matrix,
        )
        return random_strategy(iteration_data_new)
    return random_smart_strategy


