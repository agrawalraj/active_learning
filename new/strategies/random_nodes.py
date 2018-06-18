import random


def random_strategy(data, g, config):
    interventions = random.sample(g.nodes, config.max_interventions)
    n_samples = [int(config.n_samples / (config.n_batches * config.max_interventions)) for _ in interventions]
    return interventions, n_samples