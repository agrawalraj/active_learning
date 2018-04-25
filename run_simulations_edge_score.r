source('sampling.r')
source('strategies.R')
source('performance_metrics.r')
source('edge_shrink_score.r')

n_dags = 100
K = 3
M = 5
p = 20

post_given_siginv = function(siginv) {
  u = function(g, data) {
    return(unnorm_int_post_known(g, siginv, data))
  }
  return(u)
}

collect_data = function(g_star, N) {
  B = construct_B(g_star)
  p = nnodes(g_star)
  id = diag(p)
  siginv = (id - B) %*% t(id - B)
  g0 = rand_from_MEC(g_star)
  unnorm_post = function(siginv) {
    unnorm_post_known = function(g, data) {
      return(unnorm_int_post_known(g, siginv, data))
    }
    return(unnorm_post_known)
  }
  u = unnorm_post(siginv)
  n_samples = N*M*K
  collected_data = edge_shrink.strategy.simulator(K, M, n_samples, cov_edge_sampler, gen_gaus_int_data, g_star, B, g0, u)
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

# g = bnlearn::empty.graph(as.character(1:3))
# g = set.arc(g, '1', '2')
# g = set.arc(g, '1', '3')
# g = set.arc(g, '2', '3')
# res = collect_data(g, 10)
# B = res$B
# 
# id = diag(3)
# siginv = (id - B) %*% t(id - B)
# g0 = rand_from_MEC(g)
# u = post_given_siginv(siginv)
# c = class_acc(g, g0, res$data, u)

foldernum = 20

for (N in c(100)) {
  foldernum = foldernum + 1
  folder = qq('dags@{foldernum}')
  dir.create(folder, showWarnings=FALSE)
  i = 1
  while (i < n_dags) {
    filename = qq('@{folder}/dag_@{i}.RData')
    if (!file.exists(filename)) {
      print(qq('dag #@{i}'))
      print('=============================')
      
      
      g_star = bnlearn::random.graph(as.character(1:p), num=1, method='ordered', prob=5/p)
      r = covered_edges(g_star)
      n_rev = true_nrow(r)
      if (n_rev != 0) {
        vars = collect_data(g_star, N)
        save(list='vars', file=filename)
        i = i + 1
      }
    } else {
      i = i + 1
    }
  }
}
