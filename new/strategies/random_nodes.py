import random


def random_strategy(g, data, config, batch_num):
    n = config.n_samples / (config.n_batches * config.max_interventions)
    if int(n) != n:
        raise ValueError('n_samples / (n_batches * max interventions) must be an integer')
    interventions = {iv: int(n) for iv in random.sample(g.nodes, config.max_interventions)}

    return interventions