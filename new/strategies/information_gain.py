from strategies.collect_dags import collect_dags
from utils import graph_utils
import xarray as xr
import numpy as np
import itertools as itr
from collections import defaultdict
from scipy.special import logsumexp
from scipy import special
import random
from tqdm import tqdm
import operator as op


def binary_entropy(probs):
    probs = probs.copy()
    probs[probs < 0] = 0
    probs[probs > 1] = 1
    return special.entr(probs) - special.xlog1py(1 - probs, -probs)


def create_info_gain_strategy_dag_collection(dag_collection, graph_functionals, functional_entropy_fxns, gauss_iv=True, minibatch_size=10, multiplier=1, verbose=False):
    def info_gain_strategy(iteration_data):
        nsamples = iteration_data.n_samples / iteration_data.n_batches
        if int(nsamples) != nsamples:
            raise ValueError('n_samples / n_batches must be an integer')
        nsamples = int(nsamples)
        ndatapoints = 1000

        cov_mat = np.linalg.inv(iteration_data.precision_matrix)
        gauss_dags = [graph_utils.cov2dag(cov_mat, dag) for dag in dag_collection]

        if verbose:
            print('COMPUTING PRIORS')
        log_gauss_dag_weights_unnorm = np.zeros(len(dag_collection))
        prior_logpdfs = np.array([gdag.logpdf(iteration_data.current_data[-1]).sum(axis=0) for gdag in gauss_dags])
        log_gauss_dag_weights_unnorm += prior_logpdfs
        for iv_node, data in iteration_data.current_data.items():
            if iv_node != -1 and data.shape[0] != 0:
                iv_ix = iteration_data.intervention_set.index(iv_node)
                intervention = {iv_node: iteration_data.interventions[iv_ix]}
                logpdfs = np.array([gdag.logpdf(data, interventions=intervention).sum(axis=0) for gdag in gauss_dags])
                log_gauss_dag_weights_unnorm += logpdfs

        gauss_dag_weights = np.exp(log_gauss_dag_weights_unnorm - logsumexp(log_gauss_dag_weights_unnorm))
        if verbose:
            print('PRIORS:')
            for gauss_dag, weight, unnorm_w in zip(gauss_dags, gauss_dag_weights, log_gauss_dag_weights_unnorm):
                print('================')
                print(gauss_dag.arcs)
                print('prob', weight)
        if not np.isclose(gauss_dag_weights.sum(), 1):
            raise ValueError('Not correctly normalized')

        # == CREATE MATRIX MAPPING EACH GRAPH TO 0 or 1 FOR THE SPECIFIED FUNCTIONALS
        functional_matrix = np.zeros([len(dag_collection), len(graph_functionals)], dtype=int)
        for (dag_ix, dag), (functional_ix, functional) in itr.product(enumerate(gauss_dags), enumerate(graph_functionals)):
            functional_matrix[dag_ix, functional_ix] = functional(dag)

        # === FOR EACH GRAPH, OBTAIN SAMPLES FOR EACH INTERVENTION THAT'LL BE USED TO BUILD UP THE HYPOTHETICAL DATASET
        if gauss_iv:
            if verbose:
                print('COMPUTING CROSS ENTROPIES')
            cross_entropies = xr.DataArray(
                np.zeros([len(dag_collection), len(iteration_data.intervention_set), len(dag_collection)]),
                dims=['outer_dag', 'intervention_ix', 'inner_dag'],
                coords={
                    'outer_dag': list(range(len(dag_collection))),
                    'intervention_ix': list(range(len(iteration_data.interventions))),
                    'inner_dag': list(range(len(dag_collection))),
                }
            )
            for outer_dag_ix in tqdm(range(len(dag_collection)), total=len(dag_collection)):
                for intv_ix, intervention in tqdm(enumerate(iteration_data.interventions), total=len(iteration_data.interventions)):
                    for inner_dag_ix, inner_dag in enumerate(gauss_dags):
                        loc = dict(outer_dag=outer_dag_ix, intervention_ix=intv_ix, inner_dag=inner_dag_ix)
                        gdag1 = gauss_dags[outer_dag_ix]
                        gdag2 = gauss_dags[inner_dag_ix]
                        cross_entropy = graph_utils.cross_entropy_interventional(gdag1, gdag2, iteration_data.intervention_set[intv_ix], intervention.variance)
                        cross_entropies.loc[loc] = cross_entropy
        else:
            if verbose:
                print('COLLECTING DATA POINTS')
            datapoints = [
                [
                    dag.sample_interventional({intervened_node: intervention}, nsamples=ndatapoints)
                    for intervened_node, intervention in
                    zip(iteration_data.intervention_set, iteration_data.interventions)
                ]
                for dag in gauss_dags
            ]

            if verbose:
                print('CALCULATING LOG PDFS')
            datapoint_ixs = list(range(ndatapoints))
            logpdfs = xr.DataArray(
                np.zeros([len(dag_collection), len(iteration_data.intervention_set), len(dag_collection), ndatapoints]),
                dims=['outer_dag', 'intervention_ix', 'inner_dag', 'datapoint'],
                coords={
                    'outer_dag': list(range(len(dag_collection))),
                    'intervention_ix': list(range(len(iteration_data.interventions))),
                    'inner_dag': list(range(len(dag_collection))),
                    'datapoint': datapoint_ixs
                }
            )

            for outer_dag_ix in tqdm(range(len(dag_collection)), total=len(dag_collection)):
                for intv_ix, intervention in tqdm(enumerate(iteration_data.interventions),
                                                  total=len(iteration_data.interventions)):
                    for inner_dag_ix, inner_dag in enumerate(gauss_dags):
                        loc = dict(outer_dag=outer_dag_ix, intervention_ix=intv_ix, inner_dag=inner_dag_ix)
                        logpdfs.loc[loc] = inner_dag.logpdf(
                            datapoints[outer_dag_ix][intv_ix],
                            interventions={iteration_data.intervention_set[intv_ix]: intervention}
                        )
            cross_entropies = logpdfs.mean(dim='datapoint')

        current_logpdfs = np.zeros([len(dag_collection), len(dag_collection)])
        for inner_dag_ix, logpdf in enumerate(log_gauss_dag_weights_unnorm):
            current_logpdfs[:,inner_dag_ix] = logpdf

        selected_interventions = defaultdict(int)

        num_minibatches = int(np.ceil(nsamples/minibatch_size))
        mbsizes = [
            minibatch_size if m_num != num_minibatches-1 else nsamples-(num_minibatches-1)*minibatch_size
            for m_num in range(num_minibatches)
        ]

        if verbose:
            print('ALLOCATING SAMPLES')
        for minibatch_num, mbsize in tqdm(zip(range(num_minibatches), mbsizes), total=num_minibatches):
            intervention_scores = np.zeros(len(iteration_data.interventions))
            intervention_logpdfs = np.zeros([len(iteration_data.interventions), len(dag_collection), len(dag_collection)])
            nonzero_interventions = [intv_ix for intv_ix, ns in selected_interventions.items() if ns != 0]
            if iteration_data.max_interventions is not None and len(nonzero_interventions) >= iteration_data.max_interventions:
                intv_ixs_to_consider = nonzero_interventions
            else:
                intv_ixs_to_consider = range(len(iteration_data.interventions))

            for intv_ix in intv_ixs_to_consider:
                for outer_dag_ix in range(len(dag_collection)):
                    intervention_logpdfs[intv_ix, outer_dag_ix] = cross_entropies.sel(
                        outer_dag=outer_dag_ix,
                        intervention_ix=intv_ix,
                    )*mbsize*multiplier
                    new_logpdfs = current_logpdfs[outer_dag_ix] + intervention_logpdfs[intv_ix, outer_dag_ix]

                    importance_weights = np.exp(new_logpdfs - logsumexp(new_logpdfs))

                    functional_entropies = [f(functional_matrix[:, f_ix], importance_weights) for f_ix, f in enumerate(functional_entropy_fxns)]
                    intervention_scores[intv_ix] += gauss_dag_weights[outer_dag_ix] * np.sum(functional_entropies)

            best_intervention_score = intervention_scores[intv_ixs_to_consider].min()
            best_scoring_interventions = np.nonzero(intervention_scores == best_intervention_score)[0]
            best_scoring_interventions = [iv for iv in best_scoring_interventions if iv in intv_ixs_to_consider]

            selected_intv_ix = random.choice(best_scoring_interventions)
            current_logpdfs = current_logpdfs + intervention_logpdfs[selected_intv_ix]/multiplier
            selected_interventions[selected_intv_ix] += mbsize

            if verbose:
                print('SELECTED INTERVENTIONS')
                print(dict(selected_interventions))
        return selected_interventions

    return info_gain_strategy


def create_info_gain_strategy_dag_collection_enum(dag_collection, graph_functionals, functional_entropy_fxns):
    def info_gain_strategy(iteration_data):
        nsamples = iteration_data.n_samples / iteration_data.n_batches
        if int(nsamples) != nsamples:
            raise ValueError('n_samples / n_batches must be an integer')
        nsamples = int(nsamples)
        cov_mat = np.linalg.inv(iteration_data.precision_matrix)
        gauss_dags = [graph_utils.cov2dag(cov_mat, dag) for dag in dag_collection]

        log_gauss_dag_weights_unnorm = np.zeros(len(dag_collection))
        for iv_node, data in iteration_data.current_data.items():
            if iv_node == -1:
                log_gauss_dag_weights_unnorm += np.array([gdag.logpdf(data).sum(axis=0) for gdag in gauss_dags])
            else:
                iv_ix = iteration_data.intervention_set.index(iv_node)
                intervention = {iv_node: iteration_data.interventions[iv_ix]}
                print(intervention)
                log_gauss_dag_weights_unnorm += np.array([gdag.logpdf(data, interventions=intervention).sum(axis=0) for gdag in gauss_dags])
        gauss_dag_weights = np.exp(log_gauss_dag_weights_unnorm - logsumexp(log_gauss_dag_weights_unnorm))
        print(gauss_dag_weights)
        if not np.isclose(gauss_dag_weights.sum(), 1):
            raise ValueError('Not correctly normalized')

        # == CREATE MATRIX MAPPING EACH GRAPH TO 0 or 1 FOR THE SPECIFIED FUNCTIONALS
        functional_matrix = np.zeros([len(dag_collection), len(graph_functionals)], dtype=int)
        for (dag_ix, dag), (functional_ix, functional) in itr.product(enumerate(gauss_dags), enumerate(graph_functionals)):
            functional_matrix[dag_ix, functional_ix] = functional(dag)

        # === FOR EACH GRAPH, OBTAIN SAMPLES FOR EACH INTERVENTION THAT'LL BE USED TO BUILD UP THE HYPOTHETICAL DATASET
        print('COLLECTING DATA POINTS')
        datapoints = [
            [
                dag.sample_interventional({intervened_node: intervention}, nsamples=100)
                for intervened_node, intervention in
                zip(iteration_data.intervention_set, iteration_data.interventions)
            ]
            for dag in gauss_dags
        ]

        print('CALCULATING LOG PDFS')
        logpdfs = xr.DataArray(
            np.zeros([len(dag_collection), len(iteration_data.intervention_set), len(dag_collection)]),
            dims=['outer_dag', 'intervention_ix', 'inner_dag'],
            coords={
                'outer_dag': list(range(len(dag_collection))),
                'intervention_ix': list(range(len(iteration_data.interventions))),
                'inner_dag': list(range(len(dag_collection))),
            }
        )

        for outer_dag_ix in tqdm(range(len(dag_collection)), total=len(dag_collection)):
            for intv_ix, intervention in tqdm(enumerate(iteration_data.interventions), total=len(iteration_data.interventions)):
                for inner_dag_ix, inner_dag in enumerate(gauss_dags):
                    loc = dict(outer_dag=outer_dag_ix, intervention_ix=intv_ix, inner_dag=inner_dag_ix)
                    logpdfs.loc[loc] += inner_dag.logpdf(
                        datapoints[outer_dag_ix][intv_ix],
                        interventions={iteration_data.intervention_set[intv_ix]: intervention}
                    ).mean()

        starting_logpdfs = np.zeros([len(dag_collection), len(dag_collection)])
        for inner_dag_ix, logpdf in enumerate(log_gauss_dag_weights_unnorm):
            starting_logpdfs[:, inner_dag_ix] = logpdf

        combo2score = {}
        combo2selected_interventions = {}
        for combo_ix, combo in enumerate(itr.combinations(range(len(iteration_data.intervention_set)), iteration_data.max_interventions)):
            selected_interventions_combo = defaultdict(int)
            current_logpdfs = starting_logpdfs.copy()
            tmp_ix2intv_ix = dict(enumerate(combo))
            intv_ix2tmp_ix = {i_ix: t_ix for t_ix, i_ix in tmp_ix2intv_ix.items()}

            last_intervention_score = 0
            for sample_num in tqdm(range(nsamples), total=nsamples):
                intervention_scores = np.zeros(len(combo))
                intervention_logpdfs = np.zeros([len(combo), len(dag_collection), len(dag_collection)])

                for intv_ix in combo:
                    tmp_ix = intv_ix2tmp_ix[intv_ix]
                    for outer_dag_ix in range(len(dag_collection)):
                        intervention_logpdfs[tmp_ix, outer_dag_ix] = logpdfs.sel(
                            outer_dag=outer_dag_ix,
                            intervention_ix=intv_ix,
                        )

                        new_logpdfs = current_logpdfs[outer_dag_ix] + intervention_logpdfs[tmp_ix, outer_dag_ix]

                        importance_weights = np.exp(new_logpdfs - logsumexp(new_logpdfs))
                        # functional_probabilities = (importance_weights[:, np.newaxis] * functional_matrix).sum(axis=0)

                        functional_entropies = [f(functional_matrix[:, f_ix], importance_weights) for f_ix, f in
                                                enumerate(functional_entropy_fxns)]
                        # functional_entropies = binary_entropy(functional_probabilities)
                        intervention_scores[tmp_ix] += gauss_dag_weights[outer_dag_ix] * np.sum(functional_entropies)
                        # print(intervention_scores)
                best_intervention_score = intervention_scores.min()
                best_scoring_interventions = np.nonzero(intervention_scores == best_intervention_score)[0]
                selected_intv_ix = random.choice(best_scoring_interventions)
                current_logpdfs = current_logpdfs + intervention_logpdfs[selected_intv_ix]
                selected_interventions_combo[tmp_ix2intv_ix[selected_intv_ix]] += 1
                print(sample_num, best_intervention_score)

                if sample_num == nsamples - 1:
                    last_intervention_score = best_intervention_score
            combo2score[combo_ix] = last_intervention_score
            print(last_intervention_score)
            combo2selected_interventions[combo_ix] = selected_interventions_combo

        best_combo_score = min(combo2score.items(), key=op.itemgetter(1))[1]
        best_combo_ixs = [combo_ix for combo_ix, score in combo2score.items() if score == best_combo_score]
        selected_combo_ix = random.choice(best_combo_ixs)
        selected_interventions = combo2selected_interventions[selected_combo_ix]

            # nonzero_interventions = [intv_ix for intv_ix, ns in selected_interventions.items() if ns != 0]
            # if iteration_data.max_interventions is None or len(
            #         nonzero_interventions) < iteration_data.max_interventions:
            #     best_intervention_score = intervention_scores.min()
            #     best_scoring_interventions = np.nonzero(intervention_scores == best_intervention_score)[0]
            # else:
            #     best_intervention_score = intervention_scores[nonzero_interventions].min()
            #     best_scoring_interventions = np.nonzero(intervention_scores == best_intervention_score)[0]
            #     best_scoring_interventions = [iv for iv in best_scoring_interventions if iv in nonzero_interventions]
            #
            # selected_intv_ix = random.choice(best_scoring_interventions)
            # current_logpdfs = current_logpdfs + intervention_logpdfs[selected_intv_ix]
            # selected_interventions[selected_intv_ix] += 1
        return selected_interventions

    return info_gain_strategy


def create_info_gain_strategy(n_boot, graph_functionals, enum_combos=False):
    def info_gain_strategy(iteration_data):
        # === CALCULATE NUMBER OF SAMPLES IN EACH INTERVENTION
        nsamples = iteration_data.n_samples / iteration_data.n_batches
        if int(nsamples) != nsamples:
            raise ValueError('n_samples / n_batches must be an integer')
        nsamples = int(nsamples)
        cov_mat = np.linalg.inv(iteration_data.precision_matrix)
        sampled_dags = collect_dags(iteration_data.batch_folder, iteration_data.current_data, n_boot)
        gauss_dags = [graph_utils.cov2dag(cov_mat, dag) for dag in sampled_dags]

        # == CREATE MATRIX MAPPING EACH GRAPH TO 0 or 1 FOR THE SPECIFIED FUNCTIONALS
        functional_matrix = np.zeros([n_boot, len(graph_functionals)])
        for (dag_ix, dag), (functional_ix, functional) in itr.product(enumerate(gauss_dags), enumerate(graph_functionals)):
            functional_matrix[dag_ix, functional_ix] = functional(dag)

        # === FOR EACH GRAPH, OBTAIN SAMPLES FOR EACH INTERVENTION THAT'LL BE USED TO BUILD UP THE HYPOTHETICAL DATASET
        print('COLLECTING DATA POINTS')
        datapoints = [
            [
                dag.sample_interventional({intervened_node: intervention}, nsamples=nsamples)
                for intervened_node, intervention in zip(iteration_data.intervention_set, iteration_data.interventions)
            ]
            for dag in gauss_dags
        ]

        print('CALCULATING LOG PDFS')
        datapoint_ixs = list(range(nsamples))
        logpdfs = xr.DataArray(
            np.zeros([n_boot, len(iteration_data.intervention_set), n_boot, nsamples]),
            dims=['outer_dag', 'intervention_ix', 'inner_dag', 'datapoint'],
            coords={
                'outer_dag': list(range(n_boot)),
                'intervention_ix': list(range(len(iteration_data.interventions))),
                'inner_dag': list(range(n_boot)),
                'datapoint': datapoint_ixs
            }
        )
        for outer_dag_ix in tqdm(range(n_boot), total=n_boot):
            for intv_ix, intervention in tqdm(enumerate(iteration_data.interventions), total=len(iteration_data.interventions)):
                for inner_dag_ix, inner_dag in enumerate(gauss_dags):
                    loc = dict(outer_dag=outer_dag_ix, intervention_ix=intv_ix, inner_dag=inner_dag_ix)
                    logpdfs.loc[loc] = inner_dag.logpdf(
                        datapoints[outer_dag_ix][intv_ix],
                        interventions={iteration_data.intervention_set[intv_ix]: intervention}
                    )

        print('COLLECTING SAMPLES')
        if not enum_combos:
            current_logpdfs = np.zeros([n_boot, n_boot])
            selected_interventions = defaultdict(int)
            for sample_num in tqdm(range(nsamples), total=nsamples):
                intervention_scores = np.zeros(len(iteration_data.interventions))
                intervention_logpdfs = np.zeros([len(iteration_data.interventions), n_boot, n_boot])
                for intv_ix in range(len(iteration_data.interventions)):
                    # current number of times this intervention has already been selected
                    selected_datapoint_ixs = random.choices(datapoint_ixs, k=10)
                    for outer_dag_ix in range(n_boot):
                        intervention_logpdfs[intv_ix, outer_dag_ix] = logpdfs.sel(
                            outer_dag=outer_dag_ix,
                            intervention_ix=intv_ix,
                            datapoint=selected_datapoint_ixs
                        ).sum(dim='datapoint')
                        new_logpdfs = current_logpdfs[outer_dag_ix] + intervention_logpdfs[intv_ix, outer_dag_ix]

                        importance_weights = np.exp(new_logpdfs - logsumexp(new_logpdfs))
                        functional_probabilities = (importance_weights[:, np.newaxis] * functional_matrix).sum(axis=0)

                        functional_entropies = binary_entropy(functional_probabilities)
                        intervention_scores[intv_ix] += functional_entropies.sum()
                # print(intervention_scores)

                nonzero_interventions = [intv_ix for intv_ix, ns in selected_interventions.items() if ns != 0]
                if iteration_data.max_interventions is None or len(nonzero_interventions) < iteration_data.max_interventions:
                    best_intervention_score = intervention_scores.min()
                    best_scoring_interventions = np.nonzero(intervention_scores == best_intervention_score)[0]
                else:
                    best_intervention_score = intervention_scores[nonzero_interventions].min()
                    best_scoring_interventions = np.nonzero(intervention_scores == best_intervention_score)[0]
                    best_scoring_interventions = [iv for iv in best_scoring_interventions if iv in nonzero_interventions]

                selected_intv_ix = random.choice(best_scoring_interventions)
                current_logpdfs = current_logpdfs + intervention_logpdfs[selected_intv_ix]
                selected_interventions[selected_intv_ix] += 1
        else:
            combo2score = {}
            combo2selected_interventions = {}
            for combo_ix, combo in enumerate(itr.combinations(range(len(iteration_data.intervention_set)), iteration_data.max_interventions)):
                current_logpdfs = np.zeros([n_boot, n_boot])
                selected_interventions_for_combo = defaultdict(int)
                tmp_ix2intv_ix = dict(enumerate(combo))
                intv_ix2tmp_ix = {i_ix: t_ix for t_ix, i_ix in tmp_ix2intv_ix.items()}

                # for each combination of interventions, optimize greedily over samples
                for sample_num in range(nsamples):
                    intervention_scores = np.zeros(len(combo))
                    intervention_logpdfs = np.zeros([len(combo), n_boot, n_boot])

                    for intv_ix in combo:
                        # current number of times this intervention has already been selected
                        datapoint_ix = selected_interventions_for_combo[intv_ix]

                        for outer_dag_ix in range(n_boot):
                            tmp_ix = intv_ix2tmp_ix[intv_ix]
                            intervention_logpdfs[tmp_ix, outer_dag_ix] = logpdfs.sel(
                                outer_dag=outer_dag_ix,
                                intervention_ix=intv_ix,
                                datapoint=datapoint_ix
                            )
                            new_logpdfs = current_logpdfs[outer_dag_ix] + intervention_logpdfs[tmp_ix, outer_dag_ix]
                            importance_weights = np.exp(new_logpdfs - logsumexp(new_logpdfs))
                            functional_probabilities = (importance_weights[:, np.newaxis] * functional_matrix).sum(
                                axis=0)

                            functional_entropies = binary_entropy(functional_probabilities)
                            intervention_scores[tmp_ix] += functional_entropies.sum()

                    best_intervention_score = intervention_scores.min()
                    best_scoring_interventions = np.nonzero(intervention_scores == best_intervention_score)[0]
                    selected_intv_ix = random.choice(best_scoring_interventions)
                    current_logpdfs = current_logpdfs + intervention_logpdfs[selected_intv_ix]
                    selected_interventions_for_combo[tmp_ix2intv_ix[selected_intv_ix]] += 1
                combo2score[combo_ix] = intervention_scores.sum()
                combo2selected_interventions[combo_ix] = selected_interventions_for_combo
            best_combo_score = min(combo2score.items(), key=op.itemgetter(1))[1]
            best_combo_ixs = [combo_ix for combo_ix, score in combo2score.items() if score == best_combo_score]
            selected_combo_ix = random.choice(best_combo_ixs)
            selected_interventions = combo2selected_interventions[selected_combo_ix]

        return selected_interventions

    return info_gain_strategy


if __name__ == '__main__':
    import causaldag as cd
    from dataclasses import dataclass
    from typing import Any, Dict

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

    np.random.seed(100)
    g = cd.rand.directed_erdos(10, .5)
    g = cd.GaussDAG(nodes=list(range(10)), arcs=g.arcs)

    mec = [cd.DAG(arcs=arcs) for arcs in cd.DAG(arcs=g.arcs).cpdag().all_dags()]
    strat = create_info_gain_strategy_dag_collection(mec, [get_mec_functional_k(mec)], [get_k_entropy_fxn(len(mec))], verbose=True)
    samples = g.sample(1000)
    precision_matrix = samples.T @ samples / 1000
    sel_interventions = strat(
        IterationData(
            current_data={-1: g.sample(1000)},
            max_interventions=1,
            n_samples=500,
            batch_num=0,
            n_batches=1,
            intervention_set=[0, 1, 2],
            interventions=[cd.GaussIntervention() for _ in range(3)],
            batch_folder='test_sanity',
            precision_matrix=precision_matrix
        )
    )


