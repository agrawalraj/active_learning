from strategies.collect_dags import collect_dags
from utils import graph_utils


def create_info_gain_strategy(n_boot):
    def info_gain_strategy(iteration_data):
        # === CALCULATE NUMBER OF SAMPLES IN EACH INTERVENTION
        n = iteration_data.n_samples / (iteration_data.n_batches * iteration_data.max_interventions)
        if int(n) != n:
            raise ValueError('n_samples / (n_batches * max interventions) must be an integer')

        sampled_dags = collect_dags(iteration_data.batch_folder, iteration_data.current_data, n_boot)
        # TODO estimate prec from iteration_data.current_data ???
        gauss_dags = [graph_utils.prec2dag(prec, dag.topological_sort()) for dag in sampled_dags]
        # TODO take n_samples/n_batches samples for each intervention

        # TODO calculate tensor of log probabilities

    return info_gain_strategy

