from dataclasses import dataclass, asdict
import yaml
import sys

sys.path.append('../')

from utils import graph_utils
import numpy as np
import os
import causaldag as cd
from config import DATA_FOLDER
from typing import Dict, Any
import time
import itertools as itr


def get_component_dag(nnodes, p, nclusters=3):
    cluster_cutoffs = [int(nnodes/nclusters)*i for i in range(nclusters+1)]
    clusters = [list(range(cluster_cutoffs[i], cluster_cutoffs[i+1])) for i in range(len(cluster_cutoffs)-1)]
    pairs_in_clusters = [list(itr.combinations(cluster, 2)) for cluster in clusters]
    bools = np.random.binomial(1, p, sum(map(len, pairs_in_clusters)))
    dag = cd.DAG(nodes=set(range(nnodes)))
    for (i, j), b in zip(itr.chain(*pairs_in_clusters), bools):
        if b != 0:
            dag.add_arc(i, j)
    return dag



@dataclass
class GenerationConfig:
    n_nodes: int
    edge_prob: float
    n_dags: int
    graph_type: str = 'erdos'

    def save_dags(self, folder):
        os.makedirs(folder, exist_ok=True)
        yaml.dump(asdict(self), open(os.path.join(folder, 'config.yaml'), 'w'))
        if self.graph_type == 'erdos':
            dags = cd.rand.directed_erdos(self.n_nodes, self.edge_prob, size=self.n_dags)
            if self.n_dags == 1:
                dags = [dags]
        elif self.graph_type == 'components':
            dags = [get_component_dag(self.n_nodes, self.edge_prob) for _ in range(self.n_dags)]
        else:
            dags = [graph_utils.generate_DAG(self.n_nodes, type_=self.graph_type) for _ in range(self.n_dags)]
        dag_arcs = [{(i, j): 1 for i, j in dag.arcs} for dag in dags]
        gdags = [cd.GaussDAG(nodes=list(range(self.n_nodes)), arcs=arcs) for arcs in dag_arcs]

        print('=== Saving DAGs ===')
        for i, gdag in enumerate(gdags):
            os.makedirs(os.path.join(DATA_FOLDER, folder, 'dags', 'dag%d' % i), exist_ok=True)
            np.savetxt(os.path.join(DATA_FOLDER, folder, 'dags', 'dag%d' % i, 'adjacency.txt'), gdag.to_amat())
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
    target: int
    constant_intervention: bool = None

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
    precision_matrix: np.ndarray


def simulate(strategy, simulator_config, gdag, strategy_folder, num_bootstrap_dags_final=100):
    start = time.time()

    # === SAVE SIMULATION META-INFORMATION
    os.makedirs(strategy_folder, exist_ok=True)
    simulator_config.save(strategy_folder)

    # === SAMPLE SOME OBSERVATIONAL DATA TO START WITH
    n_nodes = len(gdag.nodes)
    all_samples = {i: np.zeros([0, n_nodes]) for i in range(n_nodes)}
    all_samples[-1] = gdag.sample(simulator_config.starting_samples)
    precision_matrix = np.linalg.inv(all_samples[-1].T @ all_samples[-1])

    # === GET GIES SAMPLES GIVEN JUST OBSERVATIONAL DATA
    initial_samples_path = os.path.join(strategy_folder, 'initial_samples.csv')
    initial_interventions_path = os.path.join(strategy_folder, 'initial_interventions')
    initial_gies_dags_path = os.path.join(strategy_folder, 'initial_dags/')
    graph_utils._write_data(all_samples, initial_samples_path, initial_interventions_path)
    graph_utils.run_gies_boot(num_bootstrap_dags_final, initial_samples_path, initial_interventions_path, initial_gies_dags_path)
    amats, dags = graph_utils._load_dags(initial_gies_dags_path, delete=True)
    for d, amat in enumerate(amats):
        np.save(os.path.join(initial_gies_dags_path, 'dag%d.npy' % d), amat)

    # === SPECIFY INTERVENTIONAL DISTRIBUTIONS BASED ON EACH NODE'S STANDARD DEVIATION
    intervention_set = list(range(n_nodes))
    if simulator_config.constant_intervention:
        interventions = [
            cd.BinaryIntervention(
                intervention1=cd.ConstantIntervention(val=-simulator_config.intervention_strength),
                intervention2=cd.ConstantIntervention(val=simulator_config.intervention_strength)
            ) for _ in intervention_set
        ]
    else:
        interventions = [
            cd.BinaryIntervention(
                intervention1=cd.ConstantIntervention(val=-simulator_config.intervention_strength*std),
                intervention2=cd.ConstantIntervention(val=simulator_config.intervention_strength*std)
            ) for std in np.diag(gdag.covariance)**.5
        ]

    # === RUN STRATEGY ON EACH BATCH
    for batch in range(simulator_config.n_batches):
        print('Batch %d with %s' % (batch, simulator_config))
        batch_folder = os.path.join(strategy_folder, 'dags_batch=%d/' % batch)
        os.makedirs(batch_folder, exist_ok=True)
        iteration_data = IterationData(
            current_data=all_samples,
            max_interventions=simulator_config.max_interventions,
            n_samples=simulator_config.n_samples,
            batch_num=batch,
            n_batches=simulator_config.n_batches,
            intervention_set=intervention_set,
            interventions=interventions,
            batch_folder=batch_folder,
            precision_matrix=precision_matrix
        )
        recommended_interventions = strategy(iteration_data)
        if not sum(recommended_interventions.values()) == iteration_data.n_samples / iteration_data.n_batches:
            raise ValueError('Did not return correct amount of samples')
        rec_interventions_nonzero = {intv_ix for intv_ix, ns in recommended_interventions.items() if ns != 0}
        if simulator_config.max_interventions is not None and len(rec_interventions_nonzero) > simulator_config.max_interventions:
            raise ValueError('Returned too many interventions')

        print(recommended_interventions)
        for intv_ix, nsamples in recommended_interventions.items():
            iv_node = intervention_set[intv_ix]
            new_samples = gdag.sample_interventional({iv_node: interventions[intv_ix]}, nsamples)
            all_samples[iv_node] = np.vstack((all_samples[iv_node], new_samples))

    samples_folder = os.path.join(strategy_folder, 'samples')
    os.makedirs(samples_folder, exist_ok=True)
    for i, samples in all_samples.items():
        np.savetxt(os.path.join(samples_folder, 'intervention=%d.csv' % i), samples)

    # === CHECK THE TOTAL NUMBER OF SAMPLES IS CORRECT
    nsamples_final = sum(all_samples[iv_node].shape[0] for iv_node in intervention_set + [-1])
    if nsamples_final != simulator_config.starting_samples + simulator_config.n_samples:
        raise ValueError('Did not use all samples')

    # === GET GIES SAMPLES GIVEN THE DATA FOR THIS SIMULATION
    final_samples_path = os.path.join(strategy_folder, 'final_samples.csv')
    final_interventions_path = os.path.join(strategy_folder, 'final_interventions')
    final_gies_dags_path = os.path.join(strategy_folder, 'final_dags/')
    graph_utils._write_data(all_samples, final_samples_path, final_interventions_path)
    graph_utils.run_gies_boot(num_bootstrap_dags_final, final_samples_path, final_interventions_path, final_gies_dags_path)
    amats, dags = graph_utils._load_dags(final_gies_dags_path, delete=True)
    for d, amat in enumerate(amats):
        np.save(os.path.join(final_gies_dags_path, 'dag%d.npy' % d), amat)

    with open(os.path.join(samples_folder, 'time.txt'), 'w') as f:
        f.write(str(time.time() - start))
