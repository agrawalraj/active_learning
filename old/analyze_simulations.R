library(GetoptLong)
source('sampling.r')
source('performance_metrics.r')


analyze_edge_accuracies = function(folder, ndags) {
  accuracies = c()
  for (i in 1:ndags) {
    print(qq('Calculating accuracy for DAG @{i}'))
    load(qq('@{folder}/dag_@{i}.RData'))
    g_star = vars$g_star
    B = vars$B
    p = nrow(B)
    siginv = (diag(p) - B) %*% t(diag(p) - B)
    data = vars$data
    g_seed = rand_from_MEC(g_star)
    acc = class_acc(g_star, g_seed, data, post_given_siginv(siginv), burn_in=200, thin_rate=20, niters=1000)
    accuracies = c(accuracies, acc)
  }
  save(list='accuracies', file=qq('@{folder}/accuracies.RData'))
}

analyze_approx_probs = function(folder, ndags) {
  approx_probs = c()
  for (i in 1:ndags) {
    print(qq('Calcularing approximate probability for DAG @{i}'))
    load(qq('@{folder}/dag_@{i}.RData'))
    g_star = vars$g_star
    B = vars$B
    p = nrow(B)
    siginv = (diag(p) - B) %*% t(diag(p) - B)
    data = vars$data
    approx_prob = approximate_prob_complete(g_star, data, post_given_siginv(siginv))
    print(approx_prob)
    approx_probs = c(approx_probs, approx_prob)
  }
  save(list='approx_probs', file=qq('@{folder}/approx_probs.RData'))
}

folder = 'bed_strategy/dags16'
ndags = 100
analyze_approx_probs(folder, ndags)