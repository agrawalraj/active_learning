library(GetoptLong)
source('sampling.r')
source('performance_metrics.R')

simulations_folder = 'dags9'
n_dags = 99
accuracies = c()
for (i in 1:n_dags) {
  print(qq('Calculating accuracy for DAG @{i}'))
  load(qq('@{simulations_folder}/dag_@{i}.RData'))
  g_star = vars$g_star
  B = vars$B
  p = nrow(B)
  siginv = (diag(p) - B) %*% t(diag(p) - B)
  data = vars$data
  g_seed = rand_from_MEC(g_star)
  acc = class_acc(g_star, g_seed, data, post_given_siginv(siginv), burn_in=200, thin_rate=20, niters=1000)
  accuracies = c(accuracies, acc)
}
save(list='accuracies', file=qq('@{simulations_folder}/accuracies.RData'))
