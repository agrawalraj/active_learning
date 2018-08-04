from dataclasses import dataclass
import yaml
from utils import sys_utils
from utils import graph_utils
import numpy as np
import os
import causaldag as cd
from collections import defaultdict


def get_dataset_folder(dataset_num):
    return 'data/dataset_%d/' % dataset_num


@dataclass
class SimulationConfig:
    n_samples: int
    n_batches: int
    max_interventions: int
    n_nodes: int
    edge_prob: float
    n_dags: int
    strategy: str
    intervention_strength: float
    starting_samples: int

    def save(self, dataset_num):
        params = {
            'n_samples': self.n_samples,
            'n_batches': self.n_batches,
            'max_interventions': self.max_interventions,
            'n_nodes': self.n_nodes,
            'edge_prob': self.edge_prob,
            'n_dags': self.n_dags,
            'strategy': self.strategy,
            'intervention_strength': self.intervention_strength,
            'starting_samples': self.starting_samples
        }
        filename = get_dataset_folder(dataset_num) + 'params.yaml'
        yaml.dump(params, open(filename, 'w'), indent=2, default_flow_style=False)


def simulate(strategy, config, dataset_num):
    dataset_folder = get_dataset_folder(dataset_num)
    if os.path.exists(dataset_folder):
        res = None
        while res not in ['y', 'Y', 'n', 'N']:
            res = input('Dataset %d already exists. Are you sure you want to overwrite it? ' % dataset_num)
        if res in ['n', 'N']:
            return
    sys_utils.ensure_dir(dataset_folder)
    config.save(dataset_num)

    dag_num = 0
    while dag_num < config.n_dags:
        dag_num += 1
        print('=== Simulating strategy on DAG %d ===' % dag_num)
        dag = cd.rand.graphs.directed_erdos(config.n_nodes, config.edge_prob)
        arcs = {(i, j): graph_utils.RAND_RANGE() for i, j in dag.arcs}
        gdag = cd.GaussDAG(nodes=list(range(config.n_nodes)), arcs=arcs)

        graph_folder = dataset_folder + 'graph_%d/' % dag_num
        sys_utils.ensure_dir(graph_folder)
        np.savetxt(graph_folder + 'adjacency.csv', gdag.to_amat())

        all_samples = {i: np.zeros([0, config.n_nodes]) for i in range(config.n_nodes)}
        all_samples[-1] = gdag.sample(config.starting_samples)
        for batch in range(config.n_batches):
            print('Batch %d' % batch)
            recommended_interventions = strategy(gdag, all_samples, config, batch)
            for iv, nsamples in recommended_interventions.items():
                g_iv = cd.GaussIntervention(
                    mean=config.intervention_strength,
                    variance=1
                )
                new_samples = gdag.sample_interventional({iv: g_iv}, nsamples)
                all_samples[iv] = np.vstack((all_samples[iv], new_samples))

        sys_utils.ensure_dir(graph_folder + 'samples/')
        for i, samples in all_samples.items():
            np.savetxt(graph_folder + 'samples/intervention=%d.csv' % i, samples)


if __name__ == '__main__':
    from strategies import random_nodes, learn_target_parents

    n_nodes = 10
    strategies = {
        'random': random_nodes.random_strategy,
        'learn-parents': learn_target_parents.create_learn_target_parents(n_nodes-1)
    }
    strategy = 'learn-parents'
    config = SimulationConfig(
        n_samples=100,
        n_batches=5,
        max_interventions=2,
        n_nodes=n_nodes,
        edge_prob=.5,
        n_dags=1,
        strategy=strategy,
        intervention_strength=2,
        starting_samples=250
    )

    simulate(strategies[strategy], config, 3)


