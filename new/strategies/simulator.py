from dataclasses import dataclass, asdict
import yaml
import sys

sys.path.append('../')

from utils import sys_utils
from utils import graph_utils
import numpy as np
import os
import causaldag as cd
from logger import LOGGER
from config import DATA_FOLDER
from typing import Dict, Any


@dataclass
class GenerationConfig:
    n_nodes: int
    edge_prob: float
    n_dags: int

    def save_dags(self, folder):
        os.makedirs(os.path.join(DATA_FOLDER, folder), exist_ok=True)
        yaml.dump(asdict(self), open(os.path.join(DATA_FOLDER, folder, 'config.yaml'), 'w'))
        dags = cd.rand.graphs.directed_erdos(self.n_nodes, self.edge_prob, size=self.n_dags)
        dag_arcs = [{(i, j): graph_utils.RAND_RANGE() for i, j in dag.arcs} for dag in dags]
        gdags = [cd.GaussDAG(nodes=list(range(self.n_nodes)), arcs=arcs) for arcs in dag_arcs]

        print('=== Saving DAGs ===')
        for i, gdag in enumerate(gdags):
            os.makedirs(os.path.join(DATA_FOLDER, folder, 'dag%d' % i), exist_ok=True)
            np.savetxt(os.path.join(DATA_FOLDER, folder, 'dag%d' % i, 'adjacency.txt'), gdag.to_amat())
        print('=== Saved ===')
        return gdags


@dataclass
class SimulationConfig:
    n_samples: int
    n_batches: int
    max_interventions: int
    strategy: str
    intervention_strength: float
    starting_samples: int

    def save(self, folder):
        yaml.dump(asdict(self), open(os.path.join(folder, 'sim-config.yaml'), 'w'), indent=2, default_flow_style=False)


@dataclass
class IterationData:
    current_data: Dict[Any, np.array]
    max_interventions: int
    n_samples: int
    batch_num: int
    n_batches: int
    intervention_set: list


def simulate(strategy, simulator_config, gdag, dag_folder):
    samples_folder = os.path.join(dag_folder, simulator_config.strategy, 'samples')

    # === CHECK IF IT'S OKAY TO RUN THE SIMULATIONS
    if os.path.exists(samples_folder):
        res = None
        while res not in ['y', 'Y', 'n', 'N']:
            res = input('Samples for %s already exist. Are you sure you want to overwrite it? ' % samples_folder)
        if res in ['n', 'N']:
            return

    # === SAVE SIMULATION META-INFORMATION
    os.makedirs(samples_folder, exist_ok=True)
    simulator_config.save(samples_folder)

    # === START OFF WITH OBSERVATIONAL DATA
    n_nodes = len(gdag.nodes)
    all_samples = {i: np.zeros([0, n_nodes]) for i in range(n_nodes)}
    all_samples[-1] = gdag.sample(simulator_config.starting_samples)

    # === RUN STRATEGY ON EACH BATCH
    for batch in range(simulator_config.n_batches):
        print('Batch %d' % batch)
        iteration_data = IterationData(
            current_data=all_samples,
            max_interventions=simulator_config.max_interventions,
            n_samples=simulator_config.n_samples,
            batch_num=batch,
            n_batches=simulator_config.n_batches,
            intervention_set=gdag.nodes
        )
        recommended_interventions = strategy(iteration_data)
        for iv, nsamples in recommended_interventions.items():
            g_iv = cd.GaussIntervention(
                mean=simulator_config.intervention_strength,
                variance=1
            )
            new_samples = gdag.sample_interventional({iv: g_iv}, nsamples)
            all_samples[iv] = np.vstack((all_samples[iv], new_samples))

    for i, samples in all_samples.items():
        np.savetxt(os.path.join(samples_folder, 'intervention=%d.csv' % i), samples)


if __name__ == '__main__':
    from strategies import random_nodes, learn_target_parents, edge_prob

    N_NODES = 50
    DAG_FOLDER = 'medium'
    STRATEGIES = {
        'random': random_nodes.random_strategy,
        'learn-parents': learn_target_parents.create_learn_target_parents(N_NODES - 1, 10000),
        'edge-prob': edge_prob.create_edge_prob_strategy(N_NODES-3, 10)
    }

    G_CONFIG = GenerationConfig(
        n_nodes=N_NODES,
        edge_prob=.5,
        n_dags=100
    )
    gdags = G_CONFIG.save_dags(DAG_FOLDER)

    STRATEGY = 'edge-prob'
    SIM_CONFIG = SimulationConfig(
        starting_samples=250,
        n_samples=100,
        n_batches=5,
        max_interventions=2,
        strategy=STRATEGY,
        intervention_strength=2,
    )

    for i, gdag in enumerate(gdags):
        simulate(STRATEGIES[STRATEGY], SIM_CONFIG, gdag, os.path.join(DATA_FOLDER, DAG_FOLDER, 'dag%d' % i))

