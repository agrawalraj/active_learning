source('scoring.R')
source('sampling.r')
library(GetoptLong)

#' Find the best interventions, in terms of expected number of edges oriented, given sampled dags
#' from the current posterior
#' 
#' @param essgraph The essential graph representing the Markov equivalence class (bnlearn graph)
#' @param dags sampled DAGs from the current posterior
#' @param K number of intervened nodes
#' @return set of nodes to intervene on
bed.interventions.batch <- function(essgraph, dags, K) {
  intervention.set <- c()
  for (k in 1:K) {
    intervention.scores = c()
    for (intervention in bnlearn::nodes(essgraph)) {
      if (intervention %in% intervention.set) {
        intervention.scores = c(intervention.scores, 0)
      } else {
        intervention.results = score.intervention.dags(essgraph, dags, c(intervention.set, intervention))
        intervention.score = intervention.results$score
        intervention.scores = c(intervention.scores, intervention.score)
      }
    }
    next.best.intervention <- which.max(intervention.scores)
    intervention.set = c(intervention.set, bnlearn::nodes(essgraph)[[next.best.intervention]])
  }
  return(intervention.set)
}

#' Simulate a run of the iterated BED (budgeted experimental design) policy for choosing interventions
#' and gathering data
#' 
#' @param K: number of intervened nodes per batch
#' @param M: number of batches
#' @param n_samples: total number of samples we have to allocate
#' @param sampler: function :: (g0, posterior, data, n_samples) -> [dag]
#' @param gen_int_data: function :: (g*, edge_weights, intervention_set, n_samples/intervention) -> interventional data
#' @param g_star: true DAG as bnlearn graph
#' @param B: edge weight matrix of true DAG
#' @param g0: initial dag for sampler
#' @param unnorm_post: function that takes (DAG, data) and returns unnormalized posterior probability
bed.strategy.simulator = function(K, M, n_samples, sampler, gen_int_data, g_star, B, g0, unnorm_post, sampler_niters=1000, burn_in=100, thin_rate=20) {
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
    dags = sampler(g0, u, curr_data, burn_in=burn_in, thin_rate=thin_rate, niters=sampler_niters)  # generate dags from posterior given all data
    intervention.set = bed.interventions.batch(essgraph, dags, K)  # find best interventions for this batch
    print('interventions')
    print(intervention.set)
    new_data = gen_int_data(g_star, B, as.numeric(intervention.set), rep(n_samples/(M*K), K))  # sample new data from previous intervention set
    curr_data = update_data(new_data, curr_data)  # add that data to our ongoing data set
  }
  return(curr_data)
}

#' Simulate a run of a random policy for choosing interventions and gathering data
#' 
#' @param K: number of intervened nodes per batch
#' @param M: number of batches
#' @param n_samples: total number of samples we have to allocate
#' @param g_star: true DAG as bnlearn graph
#' @param B: edge weight matrix of true DAG
#' @param gen_int_data: function :: (g*, edge_weights, intervention_set, n_samples/intervention) -> interventional data
random.strategy.simulator = function(K, M, n_samples, g_star, B, gen_int_data) {
  curr_data = list()
  curr_data = curr_data[1:bnlearn::nnodes(g_star)]
  
  for (m in 1:M) {
    print(qq('batch #@{m}'))
    intervention.set = sample.int(bnlearn::nnodes(g_star), K)
    new_data = gen_int_data(g_star, B, as.numeric(intervention.set), rep(n_samples/(M*K), K))
    curr_data = update_data(new_data, curr_data)
  }
  return(curr_data)
}






# essgraph = bnlearn::empty.graph(nodes = as.character(1:4))
# essgraph = bnlearn::set.edge(essgraph, '1', '2')
# essgraph = bnlearn::set.edge(essgraph, '1', '3')
# essgraph = bnlearn::set.edge(essgraph, '1', '4')
# essgraph = bnlearn::set.edge(essgraph, '2', '3')
# essgraph = bnlearn::set.edge(essgraph, '2', '4')
# essgraph = bnlearn::set.edge(essgraph, '3', '4')
# 
# g1 = bnlearn::empty.graph(nodes = as.character(1:4))
# g1 = bnlearn::set.arc(g1, '1', '2')
# g1 = bnlearn::set.arc(g1, '1', '3')
# g1 = bnlearn::set.arc(g1, '4', '1')
# g1 = bnlearn::set.arc(g1, '2', '3')
# g1 = bnlearn::set.arc(g1, '4', '2')
# g1 = bnlearn::set.arc(g1, '4', '3')
# 
# g2 = bnlearn::empty.graph(nodes = as.character(1:4))
# g2 = bnlearn::set.arc(g2, '1', '2')
# g2 = bnlearn::set.arc(g2, '1', '3')
# g2 = bnlearn::set.arc(g2, '4', '1')
# g2 = bnlearn::set.arc(g2, '3', '2')
# g2 = bnlearn::set.arc(g2, '4', '2')
# g2 = bnlearn::set.arc(g2, '4', '3')

# inv1 = score.intervention.dag(essgraph, g2, c('1'))
# inv2 = score.intervention.dag(essgraph, g2, c('2'))
# inv3 = score.intervention.dag(essgraph, g2, c('3'))
# inv4 = score.intervention.dag(essgraph, g2, c('4'))
# gs = list(g1, g2)
# intervention.set = bed.interventions.batch(essgraph, gs, 2)
