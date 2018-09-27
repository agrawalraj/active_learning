import argparse
import os
import numpy as np
from strategies.simulator import SimulationConfig, simulate
from strategies import random_nodes, learn_target_parents, edge_prob, var_score, information_gain
from config import DATA_FOLDER
import causaldag as cd
from multiprocessing import Pool, cpu_count

NUM_BOOTSTRAP_DAGS_BATCH = 50
NUM_STARTING_SAMPLES = 250
INTERVENTION_STRENGTH = 2

parser = argparse.ArgumentParser(description='Simulate strategy for learning parent nodes in a causal DAG.')

parser.add_argument('--samples', '-n', type=int, help='number of samples')
parser.add_argument('--batches', '-b', type=int, help='number of batches allowed')
parser.add_argument('--max_interventions', '-k', type=int, help='maximum number of interventions per batch')

parser.add_argument('--folder', type=str, help='Folder containing the DAGs')
parser.add_argument('--strategy', type=str, help='Strategy to use')

args = parser.parse_args()

ndags = len(os.listdir(os.path.join(DATA_FOLDER, args.folder, 'dags')))
amats = [np.loadtxt(os.path.join(DATA_FOLDER, args.folder, 'dags', 'dag%d' % i, 'adjacency.txt')) for i in range(ndags)]
dags = [cd.GaussDAG.from_amat(amat) for amat in amats]
nnodes = len(dags[0].nodes)
target = int(np.ceil(nnodes/2) - 1)

SIM_CONFIG = SimulationConfig(
    starting_samples=NUM_STARTING_SAMPLES,
    n_samples=args.samples,
    n_batches=args.batches,
    max_interventions=args.max_interventions,
    strategy=args.strategy,
    intervention_strength=INTERVENTION_STRENGTH,
    target=target
)


def parent_functionals(target, nodes):
    def get_parent_functional(parent):
        def parent_functional(dag):
            return parent in dag.parents[target]
        return parent_functional

    return [get_parent_functional(node) for node in nodes if node != target]


def get_strategy(strategy, dag):
    if strategy == 'random':
        return random_nodes.random_strategy
    if strategy == 'learn-parents':
        return learn_target_parents.create_learn_target_parents(target, NUM_BOOTSTRAP_DAGS_BATCH)
    if strategy == 'edge-prob':
        return edge_prob.create_edge_prob_strategy(target, NUM_BOOTSTRAP_DAGS_BATCH)
    if strategy == 'var-score':
        node_vars = np.diag(dag.covariance)
        return var_score.create_variance_strategy(target, node_vars, [2*np.sqrt(node_var) for node_var in node_vars])
    if strategy == 'entropy':
        return information_gain.create_info_gain_strategy(NUM_BOOTSTRAP_DAGS_BATCH, parent_functionals(target, dag.nodes))


folders = [
    os.path.join(DATA_FOLDER, args.folder, 'dags', 'dag%d' % i, args.strategy + ',n=%s,b=%s,k=%s' % (args.samples, args.batches, args.max_interventions))
    for i in range(ndags)
]


def simulate_(tup):
    dag, folder = tup
    simulate(get_strategy(args.strategy, dag), SIM_CONFIG, dag, folder)


with Pool(cpu_count()-1) as p:
    p.map(simulate_, zip(dags, folders))

