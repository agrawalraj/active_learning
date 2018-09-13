from strategies.collect_dags import collect_dags
from utils import graph_utils
import xarray as xr
import numpy as np
import itertools as itr


def create_info_gain_strategy(precision_matrix, n_boot):
    def info_gain_strategy(iteration_data):
        # === CALCULATE NUMBER OF SAMPLES IN EACH INTERVENTION
        nsamples = iteration_data.n_samples / (iteration_data.n_batches * iteration_data.max_interventions)
        if int(nsamples) != nsamples:
            raise ValueError('n_samples / (n_batches * max interventions) must be an integer')
        nnodes = precision_matrix.shape[0]

        sampled_dags = collect_dags(iteration_data.batch_folder, iteration_data.current_data, n_boot)
        gauss_dags = [graph_utils.prec2dag(precision_matrix, dag.topological_sort()) for dag in sampled_dags]
        dag_datasets = xr.DataArray(
            np.zeros([n_boot, nnodes, nsamples]),
            dims=['dag', 'intervened_node', 'sample'],
            coords={
                'dag': list(range(n_boot)),
                'intervened_node': list(range(nnodes)),
                'sample': list(range(nsamples))
            }
        )
        for (i, gauss_dag), p in itr.product(enumerate(gauss_dags), range(nnodes)):
            dag_datasets.loc[dict(dag=i, intervened_node=p)] = iteration_data.interventions[p].sample(nsamples)

        log_probabilities = xr.DataArray(
            np.zeros([n_boot, n_boot, nnodes, nsamples]),
            dims=['generating_dag', 'dag', 'intervened_node', 'sample'],
            coords={
                'generating_dag': list(range(n_boot)),
                'dag': list(range(n_boot)),
                'intervened_node': list(range(nnodes)),
                'sample': list(range(nsamples)),
            }
        )

        # TODO calculate tensor of log probabilities

    return info_gain_strategy

