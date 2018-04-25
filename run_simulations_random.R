source('sampling.r')
source('strategies.R')
library(GetoptLong)

n_dags = 100
K = 3
M = 5
p = 20

collect_data = function(g_star, N) {
  B = construct_B(g_star)
  n_samples = N*M*K
  collected_data = random.strategy.simulator(K, M, n_samples, g_star, B, gen_gaus_int_data)
  return(list('g_star'=g_star, 'B'=B, 'data'=collected_data))
}

foldernum = 3
for (N in c(1000)) {
  foldernum = foldernum + 1
  folder = qq('random_strategy/dags@{foldernum}')
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