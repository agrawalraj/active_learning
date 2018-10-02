import argparse
import os
import numpy as np
from strategies.simulator import SimulationConfig, simulate
from strategies import random_nodes, learn_target_parents, edge_prob, var_score, information_gain
import causaldag as cd

parser = argparse.ArgumentParser(description='Running strategy on GeneNet Weaver datasets.')

parser.add_argument('--samples', '-n', type=int, help='number of samples')
parser.add_argument('--batches', '-b', type=int, help='number of batches allowed')
parser.add_argument('--max_interventions', '-k', type=int, help='maximum number of interventions per batch')
parser.add_argument('--intervention-strength', '-s', type=float,
                    help='number of standard deviations away from mean interventions occur at')
parser.add_argument('--boot', type=int, help='number of bootstrap samples')
parser.add_argument('--intervention-type', '-i', type=str)

parser.add_argument('--folder', type=str, default="./genenet_results/", help='Folder for storing the results')
parser.add_argument('--data-file', type=str, default="../../all_data.csv", help='File containing the GeneNet Weaver data')
parser.add_argument('--strategy', type=str, help='Strategy to use')

args = parser.parse_args()


def get_data(datafile):
    raw = np.genfromtxt('all_data.csv', delimiter=',')
    raw = raw[1:,:] #remove header
    intervention_labels = raw[:,-1] 
    raw = raw[:,0:-1] #remove last column of intervention labels
    
    n_nodes = raw.shape[1]

    # sort data based on intervention, -1 is observational data
    data = {-1: raw[intervention_labels==-1, :]}
    for i in range(n_nodes)
        data[i] = raw[intervention_labels==i,:]
    
    return data, n_nodes


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
    if strategy == 'learn-parents':
        return learn_target_parents.create_learn_target_parents(target, args.boot)
    if strategy == 'edge-prob':
        return edge_prob.create_edge_prob_strategy(target, args.boot)
    if strategy == 'var-score':
        node_vars = np.diag(dag.covariance)
        return var_score.create_variance_strategy(target, node_vars, [2*np.sqrt(node_var) for node_var in node_vars])
    if strategy == 'entropy':
        mec_functional = get_mec_functional_k(dag_collection)
        functional_entropies = [get_k_entropy_fxn(len(dag_collection))]
        return information_gain.create_info_gain_strategy(args.boot, [mec_functional], functional_entropies)
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
        return information_gain.create_info_gain_strategy_dag_collection(dag_collection, [mec_functional], functional_entropies, gauss_iv)
    if strategy == 'entropy-dag-collection-enum':
        base_dag = cd.DAG(nodes=set(dag.nodes), arcs=dag.arcs)
        dag_collection = [cd.DAG(nodes=set(dag.nodes), arcs=arcs) for arcs in base_dag.cpdag().all_dags()]
        # mec_functionals = get_mec_functionals(dag_collection)
        mec_functional = get_mec_functional_k(dag_collection)

        functional_entropies = [get_k_entropy_fxn(len(dag_collection))]
        # print([m(base_dag) for m in mec_functionals])
        return information_gain.create_info_gain_strategy_dag_collection_enum(dag_collection, [mec_functional], functional_entropies)

data, n_nodes = get_data(args.data_file)
folder = os.path.join(args.folder, args.strategy + ',n=%s,b=%s,k=%s' % (args.samples, args.batches, args.max_interventions)

SIM_CONFIG = SimulationConfig(
    starting_samples=n_nodes
    n_samples=args.samples,
    n_batches=args.batches,
    max_interventions=args.max_interventions,
    strategy=args.strategy,
    intervention_strength=args.intervention_strength,
    target=target,
    intervention_type=args.intervention_type if args.intervention_type is not None else 'gauss'
)

simulate(get_strategy(args.strategy, None), SIM_CONFIG, n_nodes, folder, data)
