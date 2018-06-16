source('sampling.r')
source('strategies.R')
source('performance_metrics.r')
library(doParallel)
library(GetoptLong)

n_dags = 100
K = 5
M = 5
p = 5


collect_data = function(g_star, N) {
  B = construct_B(g_star)
  
  p = nnodes(g_star)
  id = diag(p)
  siginv = (id - B) %*% t(id - B)
  u = post_given_siginv(siginv)
  n_samples = N*M*K
  
  g0 = rand_from_MEC(g_star)
  collected_data = bed.strategy.simulator(K, M, n_samples, cov_edge_sampler, gen_gaus_int_data, g_star, B, g0, u)
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


randomDAG = function() {
  nrev = 0
  while (nrev == 0) {
    g_star = bnlearn::random.graph(as.character(1:p), num=1, method='ordered', prob=1)
    r = covered_edges(g_star)
    nrev = true_nrow(r)
  }
  return(g_star)
}


simulateDAG = function(i) {
  cat(qq('starting DAG #@{i}'), file=stdout())
  filename = qq('@{folder}/dag_@{i}.RData')
  if (!file.exists(filename)) {
    g_star = randomDAG()
    vars = collect_data(g_star, N)
    save(list='vars', file=filename)
  }
}


basefolder = 'bed_strategy'
registerDoParallel(cores=3)
foldernum = 16
sample_sizes = c(10, 20, 50)
for (N in sample_sizes) {
  foldernum = foldernum + 1
  folder = qq('@{basefolder}/dags@{foldernum}')
  dir.create(folder, showWarnings = FALSE)
  # parLapply(cluster, 1:n_dags, simulateDAG, folder=folder, N=N)
  foreach(i=1:n_dags, folder=rep(folder, n_dags), N=rep(N, n_dags)) %do% simulateDAG(i)
}

