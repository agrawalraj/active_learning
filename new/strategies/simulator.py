from dataclasses import dataclass
import yaml
from utils import sys_utils
from utils import graph_utils
import numpy as np
import os


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

    def save(self, dataset_num):
        params = {
            'n_samples': self.n_samples,
            'n_batches': self.n_batches,
            'max_interventions': self.max_interventions,
            'n_nodes': self.n_nodes,
            'edge_prob': self.edge_prob,
            'n_dags': self.n_dags,
            'strategy': self.strategy
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

    dag_num = -1
    while dag_num < config.n_dags:
        dag_num += 1
        print('=== Simulating strategy on DAG %d ===' % dag_num)
        g = graph_utils.random_graph(config.n_nodes, config.edge_prob)
        if len(graph_utils.get_covered_edges(g)) == 0:
            continue
        adj_mat = graph_utils.random_adj(g)
        siginv = graph_utils.adj2prec(adj_mat)

        graph_folder = dataset_folder + 'graph_%d/' % dag_num
        sys_utils.ensure_dir(graph_folder)
        np.savetxt(graph_folder + 'adjacency.csv', adj_mat)

        all_samples = [[] for _ in range(len(g.nodes))]
        for batch in range(config.n_batches):
            print('Batch %d' % batch)
            interventions, n_samples = strategy(g, siginv, all_samples, config)
            new_samples = graph_utils.sample_graph_int(g, adj_mat, interventions, n_samples)
            all_samples = graph_utils.concatenate_data(all_samples, new_samples)

        sys_utils.ensure_dir(graph_folder + 'samples/')
        for i in range(len(g.nodes)):
            np.savetxt(graph_folder + 'samples/intervention=%d.csv' % i, all_samples[i])


if __name__ == '__main__':
    from strategies import iterated_bed, random_nodes
    strategies = {
        'iterated_bed': iterated_bed.iterated_bed,
        'random': random_nodes.random_strategy
    }
    strategy = 'random'
    config = SimulationConfig(
        n_samples=100,
        n_batches=5,
        max_interventions=2,
        n_nodes=10,
        edge_prob=.1,
        n_dags=100,
        strategy=strategy
    )


    simulate(strategies[strategy], config, 2)


