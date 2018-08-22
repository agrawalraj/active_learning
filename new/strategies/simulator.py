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
from config import DATA_FOLDER, TEMP_INTERVENTIONS_PATH, TEMP_SAMPLES_PATH, TEMP_DAG_FOLDER
from typing import Dict, Any


@dataclass
class GenerationConfig:
    n_nodes: int
    edge_prob: float
    n_dags: int

    def save_dags(self, folder):
        os.makedirs(folder, exist_ok=True)
        yaml.dump(asdict(self), open(os.path.join(folder, 'config.yaml'), 'w'))
        dags = cd.rand.directed_erdos(self.n_nodes, self.edge_prob, size=self.n_dags)
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
    interventions: list
    batch_folder: str


def simulate(strategy, simulator_config, gdag, strategy_folder, num_bootstrap_dags_final):
    samples_folder = os.path.join(strategy_folder, 'samples')

    # === SAVE SIMULATION META-INFORMATION
    os.makedirs(samples_folder, exist_ok=True)
    simulator_config.save(samples_folder)

    # === START OFF WITH OBSERVATIONAL DATA
    n_nodes = len(gdag.nodes)
    all_samples = {i: np.zeros([0, n_nodes]) for i in range(n_nodes)}
    all_samples[-1] = gdag.sample(simulator_config.starting_samples)

    # === SPECIFY INTERVENTIONAL DISTRIBUTIONS BASED ON EACH NODE'S STANDARD DEVIATION
    interventions = [
        cd.BinaryIntervention(
            intervention1=cd.ConstantIntervention(val=-simulator_config.intervention_strength*std).sample,
            intervention2=cd.ConstantIntervention(val=simulator_config.intervention_strength*std).sample
        ) for std in np.diag(gdag.covariance)**.5
    ]

    # === RUN STRATEGY ON EACH BATCH
    for batch in range(simulator_config.n_batches):
        print('Batch %d' % batch)
        batch_folder = os.path.join(strategy_folder, 'dags_batch=%d/' % batch)
        os.makedirs(batch_folder, exist_ok=True)
        iteration_data = IterationData(
            current_data=all_samples,
            max_interventions=simulator_config.max_interventions,
            n_samples=simulator_config.n_samples,
            batch_num=batch,
            n_batches=simulator_config.n_batches,
            intervention_set=gdag.nodes,
            interventions=interventions,
            batch_folder=batch_folder
        )
        recommended_interventions = strategy(iteration_data)
        for iv_node, nsamples in recommended_interventions.items():
            intervention = interventions[iv_node]
            new_samples = gdag.sample_interventional({iv_node: intervention.sample}, nsamples)
            all_samples[iv_node] = np.vstack((all_samples[iv_node], new_samples))

    for i, samples in all_samples.items():
        np.savetxt(os.path.join(samples_folder, 'intervention=%d.csv' % i), samples)

    # === GET GIES SAMPLES GIVEN THE DATA FOR THIS SIMULATION
    graph_utils._write_data(all_samples)
    final_gies_dags_path = os.path.join(strategy_folder, 'dags_final')
    graph_utils.run_gies_boot(100, TEMP_SAMPLES_PATH, TEMP_INTERVENTIONS_PATH, final_gies_dags_path)

