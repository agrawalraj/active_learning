import random


def random_strategy(g, siginv, data, config):
    interventions = random.sample(g.nodes, config.max_interventions)
    n = config.n_samples / (config.n_batches * config.max_interventions)
    if int(n) != n:
        raise ValueError('n_samples / (n_batches * max interventions) must be an integer')
    n_samples = [int(n) for _ in interventions]
    return interventions, n_samples