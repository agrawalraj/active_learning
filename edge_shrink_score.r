library(pcalg);
library(bnlearn);
library(graph);
source('scoring.R')
source('sampling.r')

# p = 20
# g = random.graph(as.character(1:p), num=1, 
#                 method="ordered", prob=4/(p - 1))

# essgraph = cpdag(g)
edge_shrinkage_score = function(essgraph, intervened_nodes, samp_dags1, samp_dags2, verbose=FALSE) {
  # Get set of resulting directed edges after intervening on 'intervened_nodes' for each graph in samp_dags1
  num_mc_samps = length(samp_dags1)
  undir_edges = undirected.arcs(essgraph)
  int_learned_arcs = lapply(samp_dags1, function(g) directed.arcs(score.intervention.dag(essgraph, g, intervened_nodes)$updated))
  
  # Cache computation of edge probabilities for later
  adj_mats = lapply(samp_dags2, function(g) amat(g))
  weights = 1 / length(samp_dags2) * rep(1, length(samp_dags2))
  p = nnodes(samp_dags2[[1]])
  avg_adj_mat = matrix(0, p, p) # edge probabilities estimated from samp_dags2
  for (i in 1:length(samp_dags2)) {
    avg_adj_mat = avg_adj_mat + weights[i] * adj_mats[[i]]
  }
  
  score = 0
  for (i in 1:nrow(undir_edges)){ # if undir_edges contains i--> j, it also contains j--->i 
    dir_edge = unname(undir_edges[i, ]) 
    edge_prob = avg_adj_mat[as.numeric(dir_edge[1]), as.numeric(dir_edge[2])]
    shrink_edge = 2 * min(edge_prob, 1 - edge_prob)
    num_times_oriented = sum(sapply(int_learned_arcs, function(dir_arc_set) sum(!colSums(t(dir_arc_set) != dir_edge)) == 1)) # really wierd stack-overflow solution fix...  
    score = score + shrink_edge * num_times_oriented
  }
  return(score / num_mc_samps)
}

edge_shrink.interventions.batch <- function(essgraph, samp_dags1, samp_dags2, K) {
  # do Monte Carlo maximization of O_p(I) over all intervention sets I
  # with less than K nodes
  intervention.set <- c()
  for (k in 1:K) {
    intervention.scores = c()
    for (intervention in bnlearn::nodes(essgraph)) {
      if (intervention %in% intervention.set) {
        intervention.scores = c(intervention.scores, 0)
      } else {
        intervention.score = edge_shrinkage_score(essgraph, samp_dags1,samp_dags2, c(intervention.set, intervention))
        intervention.scores = c(intervention.scores, intervention.score)
      }
    }
    next.best.intervention <- which.max(intervention.scores)
    intervention.set = c(intervention.set, bnlearn::nodes(essgraph)[[next.best.intervention]])
  }
  return(intervention.set)
}

edge_shrink.strategy.simulator = function(K, M, n_samples, sampler, gen_int_data, g_star, edge_weights, g0, unnorm_post, sampler_niters=1000, burn_in=100, thin_rate=20) {
  # K: number of intervened nodes per batch
  # M: number of batches
  # n_samples: total number of samples we have to allocate
  # sampler: function that takes (g0, posterior, data, n_samples) and returns ([dag])
  # gen_int_data: function that takes (g*, edge_weights, intervention_set, n_samples/intervention) and returns data
  # g_star: true DAG as bnlearn graph
  # edge_weights: edge weight matrix of true DAG
  # g0: initial dag for sampler
  # p: function that takes (DAG, data) and returns unnormalized posterior probability
  
  essgraph = bnlearn::cpdag(g_star)
  curr_data = list()
  curr_data = curr_data[1:bnlearn::nnodes(essgraph)]
  
  for (m in 1:M) {
    print(qq('batch #@{m}'))
    if (m==1) {
      u = MEC_unif_dist
    } else {
      u = unnorm_post
    }
    dags1 = sampler(g0, u, curr_data, burn_in=burn_in, thin_rate=thin_rate, niters=sampler_niters)  # generate dags from posterior given all data
    dags2 = sampler(g0, u, curr_data, burn_in=burn_in, thin_rate=thin_rate, niters=sampler_niters)  # generate dags from posterior given all data
    intervention.set = edge_shrink.interventions.batch(essgraph, dags1, dags2, K)  # find best interventions for this batch
    print('interventions')
    print(intervention.set)
    new_data = gen_int_data(g_star, edge_weights, as.numeric(intervention.set), rep(n_samples/(M*K), K))  # sample new data from previous intervention set
    curr_data = update_data(new_data, curr_data)  # add that data to our ongoing data set
  }
  return(curr_data)
}
