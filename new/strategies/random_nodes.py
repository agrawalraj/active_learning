import random
from collections import Counter


def random_strategy(iteration_data):
    if iteration_data.max_interventions is None:
        n = iteration_data.n_samples / iteration_data.n_batches
        if int(n) != n:
            raise ValueError('n_samples / n_batches must be an integer')
        intv_ixs = list(range(len(iteration_data.intervention_set)))
        return dict(Counter(random.choices(intv_ixs, n)))
    else:
        n = iteration_data.n_samples / (iteration_data.n_batches * iteration_data.max_interventions)
        if int(n) != n:
            raise ValueError('n_samples / (n_batches * max interventions) must be an integer')
        intv_ixs = list(range(len(iteration_data.intervention_set)))
        interventions = {intv_ix: int(n) for intv_ix in random.sample(intv_ixs, iteration_data.max_interventions)}

        return interventions


