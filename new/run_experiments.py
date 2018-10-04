import argparse
import os
import numpy as np
from strategies.simulator import SimulationConfig, simulate
from strategies import random_nodes, learn_target_parents, edge_prob, var_score, information_gain
from config import DATA_FOLDER
import causaldag as cd
from multiprocessing import Pool, cpu_count
import random

np.random.seed(1729)
random.seed(1729)

NUM_STARTING_SAMPLES = 10000

parser = argparse.ArgumentParser(description='Simulate strategy for learning parent nodes in a causal DAG.')

parser.add_argument('--samples', '-n', type=int, help='number of samples')
parser.add_argument('--batches', '-b', type=int, help='number of batches allowed')
parser.add_argument('--max_interventions', '-k', type=int, help='maximum number of interventions per batch')
parser.add_argument('--intervention-strength', '-s', type=float,
                    help='number of standard deviations away from mean interventions occur at')
parser.add_argument('--boot', type=int, help='number of bootstrap samples')
parser.add_argument('--intervention-type', '-i', type=str)
parser.add_argument('--mbsize', '-m', type=int, help='Minibatch size')
parser.add_argument('--verbose', '-v', type=bool, help='Minibatch size')

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
    intervention_strength=args.intervention_strength,
    target=target,
    intervention_type=args.intervention_type if args.intervention_type is not None else 'gauss'
)


def parent_functionals(target, nodes):
    def get_parent_functional(parent):
        def parent_functional(dag):
            return parent in dag.parents[target]
        return parent_functional

    return [get_parent_functional(node) for node in nodes if node != target]


def get_mec_functionals(dag_collection):
    def get_isdag_functional(dag):
        def isdag_functional(test_dag):
            return dag.arcs == test_dag.arcs
        return isdag_functional
    return [get_isdag_functional(dag) for dag in dag_collection]


def get_mec_functional_k(dag_collection):
    def get_dag_ix_mec(dag):
        return next(d_ix for d_ix, d in enumerate(dag_collection) if d.arcs == dag.arcs)
    return get_dag_ix_mec


def get_k_entropy_fxn(k):
    def get_k_entropy(fvals, weights):
        # find probs
        probs = np.zeros(k)
        for fval, w in zip(fvals, weights):
            probs[fval] += w

        # = find entropy
        mask = probs != 0
        plogps = np.zeros(len(probs))
        plogps[mask] = np.log2(probs[mask]) * probs[mask]
        return -plogps.sum()

    return get_k_entropy


def get_strategy(strategy, dag):
    if strategy == 'random':
        return random_nodes.random_strategy
    if strategy == 'random-smart':
        d = cd.DAG(nodes=set(dag.nodes), arcs=dag.arcs)
        return random_nodes.create_random_smart_strategy(d.cpdag())
    if strategy == 'learn-parents':
        return learn_target_parents.create_learn_target_parents(target, args.boot)
    if strategy == 'edge-prob':
        return edge_prob.create_edge_prob_strategy(target, args.boot)
    if strategy == 'var-score':
        node_vars = np.diag(dag.covariance)
        return var_score.create_variance_strategy(target, node_vars, [2*np.sqrt(node_var) for node_var in node_vars])
    if strategy == 'entropy':
        return information_gain.create_info_gain_strategy(args.boot, parent_functionals(target, dag.nodes))
    if strategy == 'entropy-enum':
        return information_gain.create_info_gain_strategy(args.boot, parent_functionals(target, dag.nodes), enum_combos=True)
    if strategy == 'entropy-dag-collection':
        base_dag = cd.DAG(nodes=set(dag.nodes), arcs=dag.arcs)
        dag_collection = [cd.DAG(nodes=set(dag.nodes), arcs=arcs) for arcs in base_dag.cpdag().all_dags()]
        # mec_functionals = get_mec_functionals(dag_collection)
        mec_functional = get_mec_functional_k(dag_collection)
        functional_entropies = [get_k_entropy_fxn(len(dag_collection))]
        # print([m(base_dag) for m in mec_functionals])

        gauss_iv = args.intervention_type == 'gauss'
        return information_gain.create_info_gain_strategy_dag_collection(dag_collection, [mec_functional], functional_entropies, gauss_iv, args.mbsize, verbose=args.verbose)
    if strategy == 'entropy-dag-collection-enum':
        base_dag = cd.DAG(nodes=set(dag.nodes), arcs=dag.arcs)
        dag_collection = [cd.DAG(nodes=set(dag.nodes), arcs=arcs) for arcs in base_dag.cpdag().all_dags()]
        # mec_functionals = get_mec_functionals(dag_collection)
        mec_functional = get_mec_functional_k(dag_collection)

        functional_entropies = [get_k_entropy_fxn(len(dag_collection))]
        # print([m(base_dag) for m in mec_functionals])
        return information_gain.create_info_gain_strategy_dag_collection_enum(dag_collection, [mec_functional], functional_entropies)


folders = [
    os.path.join(DATA_FOLDER, args.folder, 'dags', 'dag%d' % i, args.strategy + ',n=%s,b=%s,k=%s' % (args.samples, args.batches, args.max_interventions))
    for i in range(ndags)
]


def simulate_(tup):
    gdag, folder = tup
    dag = cd.DAG(nodes=set(gdag.nodes), arcs=gdag.arcs)
    print('SIMULATING FOR DAG:')
    print('Size of MEC:', len(dag.cpdag().all_dags()))
    simulate(get_strategy(args.strategy, gdag), SIM_CONFIG, gdag, folder)


with Pool(cpu_count()-1) as p:
    p.map(simulate_, zip(dags, folders))

