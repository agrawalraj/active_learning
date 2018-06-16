source('sampling.r')
library(bnlearn)

# uniform
N = 5
p = 5
g_star = bnlearn::random.graph(as.character(1:p), prob=1)
B = construct_B(g_star)
id = diag(p)
siginv = (id - B) %*% t(id - B)
u = post_given_siginv(siginv)
for (n in 1:N) {
  print(n)
  d = rand_from_MEC(g_star)
  no_data = list()
  no_data = no_data[1:bnlearn::nnodes(g_star)]
  samples = cov_edge_sampler(d, MEC_unif_dist, no_data)
  log_probs = c()
  for (s in samples) {
    log_prob = u(s, no_data)
    log_probs = c(log_probs, log_prob)
  }
  plot(log_probs)
}