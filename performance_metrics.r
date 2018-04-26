
library(pROC)
library(bnlearn)
source('sampling.r')

# Think ROC actually does not make sense to look at.....don't use this function
plot_roc_curve = function(g_star, g_seed, data, P=unnorm_int_post_known, burn_in=200, thin_rate=20, niters=1000, print.auc=FALSE, add=FALSE, col='black', lty='solid'){
  non_comp_edges = apply(as.matrix(reversible.arcs(g_star)), 2, as.numeric)
  non_comp_edges_rev = non_comp_edges[, c('to', 'from')]
  samp_dags = cov_edge_sampler(g_seed, P, data, burn_in=burn_in, thin_rate=thin_rate, niters=niters)
  adj_mats = lapply(samp_dags, function(g) amat(g))
  weights = 1/ length(samp_dags) * rep(1, length(samp_dags))
  p = nnodes(g_star)
  avg_adj_mat = matrix(0, p, p)
  for (i in 1:length(samp_dags)) {
    avg_adj_mat = avg_adj_mat + weights[i] * adj_mats[[i]]
  }
  true_adj_mat = amat(g_star)
  y_true = c(rep(1, nrow(non_comp_edges)), rep(0, nrow(non_comp_edges_rev)))
  preds = c(avg_adj_mat[non_comp_edges], avg_adj_mat[non_comp_edges_rev])
  roc_obj = roc(y_true, preds)
  plot.roc(roc_obj, print.auc=print.auc, add=add, col=col,
           xlab='True Negative Rate', ylab='True Postitive Rate', lty=lty, cex=2.5, lwd=2)
  return(roc_obj)
}

class_acc = function(g_star, g_seed, data, P=unnorm_int_post_known, burn_in=200, thin_rate=20, niters=1000){
  if (class(reversible.arcs(g_star)) == "character"){ # only one single compelled edge
    non_comp_edges = as.numeric(reversible.arcs(g_star))
    non_comp_edges = matrix(non_comp_edges, ncol=2)
  } else {
    non_comp_edges = apply(as.matrix(reversible.arcs(g_star)), 2, as.numeric)
  }
  samp_dags = cov_edge_sampler(g_seed, P, data, burn_in=burn_in, thin_rate=thin_rate, niters=niters)
  adj_mats = lapply(samp_dags, function(g) amat(g))
  weights = 1/ length(samp_dags) * rep(1, length(samp_dags))
  p = nnodes(g_star)
  avg_adj_mat = matrix(0, p, p)
  for (i in 1:length(samp_dags)) {
    avg_adj_mat = avg_adj_mat + weights[i] * adj_mats[[i]]
  }
  preds = c(avg_adj_mat[non_comp_edges])
  return(mean(preds > .5))
}

# REMOVE THIS FUNCTION 
class_preds = function(g_star, g_seed, data, P=unnorm_int_post_known, burn_in=200, thin_rate=20, niters=1000){
  if (class(reversible.arcs(g_star)) == "character"){ # only one single compelled edge
    non_comp_edges = as.numeric(reversible.arcs(g_star))
    non_comp_edges = matrix(non_comp_edges, ncol=2)
  } else {
    non_comp_edges = apply(as.matrix(reversible.arcs(g_star)), 2, as.numeric)
  }
  samp_dags = cov_edge_sampler(g_seed, P, data, burn_in=burn_in, thin_rate=thin_rate, niters=niters)
  adj_mats = lapply(samp_dags, function(g) amat(g))
  weights = 1/ length(samp_dags) * rep(1, length(samp_dags))
  p = nnodes(g_star)
  avg_adj_mat = matrix(0, p, p)
  for (i in 1:length(samp_dags)) {
    avg_adj_mat = avg_adj_mat + weights[i] * adj_mats[[i]]
  }
  return(avg_adj_mat)
}

avg_adj_mat = function(dags) {
  rev_arc_mat = as.matrix(reversible.arcs(dags[[1]]))
  non_comp_edges = apply(rev_arc_mat, 2, as.numeric)
  adj_mats = lapply(dags, function(g) amat(g))
  avg_adj_mat = matrix(0, p, p)
  for (adj_mat in adj_mats) {
    avg_adj_mat = avg_adj_mat + 1/length(dags) * adj_mat
  }
  return(avg_adj_mat[non_comp_edges])
}




