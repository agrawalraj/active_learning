library(pcalg);
library(bnlearn);
library(graph);
source('scoring.R')

g = random.graph(as.character(1:p), num=1, 
                 method="ordered", prob=4/(p - 1))

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
    shrink_edge = min(edge_prob, 1 - edge_prob)
    num_times_oriented = sum(sapply(int_learned_arcs, function(dir_arc_set) sum(!colSums(t(dir_arc_set) != dir_edge)) == 1)) # really wierd stack-overflow solution fix...  
    score = score + shrink_edge * num_times_oriented
  }
  return(score / num_mc_samps)
}

edge_shrink.interventions.batch <- function(essgraph, samp_dags1, samp_dags2, K) {
  # do Monte Carlo maximization of O_p(I) over all intervention sets I
  # with less than K nodes
  intervention.set <- list()
  for (k in 1:K) {
    intervention.scores = c()
    for (intervention in bnlearn::nodes(essgraph)) {
      if (intervention %in% intervention.set) {
        intervention.scores = c(intervention.set, 0)
      } else {
        intervention.score = edge_shrinkage_score(essgraph, samp_dags1,samp_dags2, c(intervention.set, intervention))
        print(intervention.score)
        intervention.scores = c(intervention.scores, intervention.score)
      }
    }
    next.best.intervention <- which.max(intervention.scores)
    intervention.set = c(intervention.set, bnlearn::nodes(essgraph)[[next.best.intervention]])
  }
  return(intervention.set)
}



