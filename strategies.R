source('scoring.R')
source('sampling.r')
library(GetoptLong)

# possible improvement: keep track of updated essgraph for each dag

bed.interventions.batch <- function(essgraph, dags, K) {
  # do Monte Carlo maximization of O_p(I) over all intervention sets I
  # with less than K nodes
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

bed.strategy.simulator = function(K, M, n_samples, sampler, gen_int_data, g_star, edge_weights, g0, unnorm_post, sampler_niters=1000, burn_in=100, thin_rate=20) {
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
  curr_data = curr_data[1:length(bnlearn::nodes(essgraph))]

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
    new_data = gen_int_data(g_star, edge_weights, as.numeric(intervention.set), rep(n_samples/(M*K), K))  # sample new data from previous intervention set
    curr_data = update_data(new_data, curr_data)  # add that data to our ongoing data set
  }
  return(curr_data)
}

post_given_siginv = function(siginv) {
  u = function(g, data) {
    return(unnorm_int_post_known(g, sig_inv, data))
  }
  return(u)
}


K = 2
M = 4
N = 10000
p = 4

collect_data = function(g_star) {
  B = construct_B(g_star)
  id = diag(p)
  sig_inv = t(id - B) %*% (id - B)
  g0 = rand_from_MEC(g_star)
  unnorm_post = function(sig_inv) {
    unnorm_post_known = function(g, data) {
      return(unnorm_int_post_known(g, sig_inv, data))
    }
    return(unnorm_post_known)
  }
  u = unnorm_post(sig_inv)
  collected_data = bed.strategy.simulator(K, M, N, cov_edge_sampler, gen_gaus_int_data, g_star, B, g0, u)
  return(list('g_star'=g_star, 'B'=B, 'data'=collected_data))
}

true_nrow = function(mat) {
  n = nrow(mat)
  if (!(is.null(n))) {
    return(n)
  } else {
    return(1)
  }
}

n_dags = 10
folder = 'dags3'
dir.create(folder, showWarnings=FALSE)
i = 1
while (i < n_dags) {
  print(qq('dag #@{i}'))
  print('=============================')
  g_star = bnlearn::random.graph(as.character(1:p), num=1, method='ordered', prob=.5)
  r = reversible.arcs(g_star)
  n_rev = true_nrow(r)
  if (n_rev != 0) {
    vars = collect_data(g_star)
    save(list='vars', file=qq('@{folder}/dag_@{i}.RData'))
    i = i + 1
  }
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
