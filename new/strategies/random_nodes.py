import random


def random_strategy(iteration_data):
    n = iteration_data.n_samples / (iteration_data.n_batches * iteration_data.max_interventions)
    if int(n) != n:
        raise ValueError('n_samples / (n_batches * max interventions) must be an integer')
    interventions = {iv: int(n) for iv in random.sample(iteration_data.nodes, iteration_data.max_interventions)}

    return interventions


