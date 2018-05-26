library(Matrix)
library(bnlearn)
library(sets)
library(mvtnorm)
library(KFAS)


############### FUNCTION THAT ACTUALLY MATTER ############### 
# construct_B - contruct random edge weights for a DAG  
# gen_gaus_int_data - generate interventional data from true DAG
# update_data - helper function to combine interventional data together for downstream posterior calculations
# unnorm_int_post_known - calculate P(g | D, sig_inv), where g is graph, D is interventional data, sig_inv is true precision matrix
# cov_edge_sampler - sample from the posterior via MCMC
############################################################ 

#' Generate random edge weights for DAG g from mixture .5unif(-1, .25) + .5unif(.25, 1)
#' 
#' @param g : bnlearn graph
#' @return B : adjacency matrix, with B_ij = weight from i to j
construct_B = function(g){
  p = nnodes(g)
  B = matrix(0, nrow=p, ncol=p)
  edges = apply(arcs(g), 2, function(nodes) as.numeric(nodes))
  num_edges = narcs(g)
  for(i in 1:num_edges){
    if (num_edges > 1) {
      x1 = edges[i, 1]
      x2 = edges[i, 2]
    } else {
      x1 = edges[1]
      x2 = edges[2]
    }
    z = sample(c(0, 1), 1)
    B[x1, x2] = z * runif(n=1, min = 0.25, max = 1) + (1 - z) * runif(n=1, min = -1, max = -.25) 
  }
  return(B)
}


#' Generate n samples from the graph g_star with adjacency matrix B
#' 
#' @param g_star bnlearn graph
#' @param B adjacency matrix
#' @param n number of data points to generate
#' @param sig noise variance
gen_gaus_data = function(g_star, B, n, sig=1){
  p = nnodes(g_star)
  Omega = sig * diag(rep(1, p))
  id = diag(rep(1, p))
  Sigma = solve(t(id - B)) %*% Omega %*% solve(id - B)
  return(mvtnorm::rmvnorm(n, sigma = Sigma))
}

#' @param g_star - true DAG, blearn object
#' @param params - edge weights of g_star, p x p matrix
#' @param int_set - vector of single node interventions
#' @param samp_set - vector of samples for each interventions in int_set
#' @return : list where the ith element corresponds to data collected for an 
#' intervention at i and number of samples = samp_set[i]
gen_gaus_int_data = function(g_star, B, int_set, samp_set, sig=1){
  B_interventional = B
  all_int_data = list()
  p = nnodes(g_star)
  all_int_data[[p]] = NULL # make sure right length, assume single node interventions
  for(i in 1:length(int_set)){
    int_node = int_set[i]
    B_interventional[int_node, ] = 0 # all child edges of intervened node set to 0
    B_interventional[, int_node] = 0 # all parent edges of intervened node set to 0
    int_i_data = gen_gaus_data(g_star, B_interventional, samp_set[i], sig)
    int_i_data[, int_node] = 0 
    all_int_data[[int_node]] = int_i_data
    colnames(all_int_data[[int_node]]) = as.character(1:p)
    B_interventional = B
  }
  return(all_int_data)
}

#' @param curr_data - list of interventional data returned by 'gen_gaus_int_data'
#' @param prev_data - list of interventional data returned by 'gen_gaus_int_data'
#' @return : new list with curr_data and prev_data combined
update_data = function(curr_data, prev_data){
  num_ints = length(prev_data)
  for (i in 1:num_ints){
    if(class(prev_data[[i]]) == 'matrix'){ # no int data is specified by a 0 
      curr_data[[i]] = rbind(prev_data[[i]], curr_data[[i]])
    } 
  }
  return(curr_data)
}

inc2dag = function(inc_mat){
  e = empty.graph(colnames(inc_mat))
  for(i in 1:nrow(inc_mat)){
    for(j in 1:ncol(inc_mat)){
      if (inc_mat[i, j] == 1){
        x1 = rownames(inc_mat)[i]
        x2 = colnames(inc_mat)[j]
        e = set.arc(e, x1, x2)
      }
    }
  }
  return(e)
}

single_int_dag = function(g, i){
  amat_g = amat(g)
  amat_g = amat_g[, -i]
  amat_g = amat_g[-i, ]
  return(inc2dag(amat_g))
}

unnorm_int_post = function(g, data){
  log_prob = 0 
  for(i in 1:nnodes(g)){
    if(class(data[[i]]) == 'matrix'){
      int_dag = single_int_dag(g, i) # remove all edges of node i 
      int_data = data[[i]]
      int_data = int_data[, -i] # remove node i for bge score
      log_prob = log_prob + score(int_dag, as.data.frame(int_data), type='bge') # remove node i
    }
  }
  return(log_prob)
}

#' @param g - bnlearn graph 
#' @param sig_inv - true inverse covariance matrix 
#' @param data - interventional data in the form of a list returned by 'gen_gaus_int_data'
#' @return - the (unormalzied) posterior probability of g i.e. P(G | data, sig_inv)
unnorm_int_post_known = function(g, siginv, data){
  siginv = unname(siginv)
  perm = node.ordering(g)
  g_params = gauss_params(siginv, as.numeric(perm))
  B_g = g_params[[1]] # edge weights
  p = length(perm)
  Omega = solve(g_params[[2]]) # error variances
  log_prob = 0 
  for(i in 1:nnodes(g)){
    if(class(data[[i]]) == 'matrix'){
      B_g_int = B_g[,-i]
      B_g_int = B_g_int[-i, ]
      Omega_int = Omega[, -i]
      Omega_int = Omega_int[-i, ]
      id = diag(p - 1)
      int_cov_mat = solve(t(id - B_g_int)) %*% Omega_int %*% solve(id - B_g_int)
      int_data = data[[i]]
      int_data = int_data[, -i] # remove node i for bge score
      log_prob = log_prob + sum(dmvnorm(unname(int_data), sigma=int_cov_mat, log=TRUE))
    }
  }
  return(log_prob)
}

#' @param siginv : inverse covariance matrix
#' @return u : the function from (g, data) to the unnormalized posterior
post_given_siginv = function(siginv) {
  u = function(g, data) {
    return(unnorm_int_post_known(g, siginv, data))
  }
  return(u)
}

true_nrow = function(mat) {
  n = nrow(mat)
  if (!(is.null(n))) {
    return(n)
  } else {
    return(1)
  }
}

covered_edges = function(g){
  poten_edges = reversible.arcs(g)
  n_edges = true_nrow(poten_edges)

  if (n_edges != 0) {
    covered_edges = c()
    for(i in 1:n_edges){
      if (n_edges > 1) {
        x1 = poten_edges[i, 1]
        x2 = poten_edges[i, 2]
      } else {
        x1 = poten_edges[1]
        x2 = poten_edges[2]
      }
      
      pa_x1 = parents(g, x1)
      pa_x2 = parents(g, x2)
      set_checker = union(pa_x1, x1)
      if(as.set(set_checker) == as.set(pa_x2)){
        covered_edges = c(covered_edges, x1, x2)
      }
    }
    if (!is.null(covered_edges)) {
      return(matrix(covered_edges, ncol = 2, byrow = TRUE))
    } else {
      return(matrix(nrow=0, ncol=2))
    }
  } else {
    return(matrix(nrow=0, ncol=2))
  }
}

MEC_unif_dist =  function(g, data){
  return(1)
}

#' @param g0 - bnlearn object - in MEC of true DAG
#' @param P - unormalized posterior that takes in data see 'unnorm_int_post_known'
#' @param data - interventional data collected in the form of ....
#' @return set of DAGs sampled from P
cov_edge_sampler = function(g0, P, data, burn_in=100, thin_rate=20, niters=1000){
  gt = g0  
  cov_edges_t = covered_edges(g0)
  samp_dags = list()
  for (i in 1:niters){
    # pick an edge uniformly from cov_edges_t
    num_cov_edges_t = nrow(cov_edges_t)
    rand_ind = sample(num_cov_edges_t, 1)
    gt2 = gt # make copy of gt
    gt2 = reverse.arc(gt2, cov_edges_t[rand_ind, 1], 
                      cov_edges_t[rand_ind, 2])
    cov_edges_t2 = covered_edges(gt2)
    accept_prob = min(1, 
                      (nrow(cov_edges_t) / nrow(cov_edges_t2)) 
                      * exp(P(gt2, data) - P(gt, data))) 
    z = rbinom(n=1, size=1, prob=accept_prob)
    if(z == 1){ # accept new dag 
      gt = gt2
      cov_edges_t = cov_edges_t2
    }
    samp_dags[[i]] = gt
  }
  return(samp_dags[seq(burn_in, niters, thin_rate)])
}

rand_from_MEC = function(g_star, n_samples=1, burn_in=100, thin_rate=20) {
  no_data = list()
  no_data = no_data[1:bnlearn::nnodes(g_star)]
  niters = burn_in + thin_rate * n_samples
  dags = cov_edge_sampler(g_star, MEC_unif_dist, no_data, niters=niters)
  if (n_samples==1) {
    return(dags[[1]])
  } else {
    return(dags)
  }
}

#' Get the adjacency matrix and noise variance matrix
#' corresponding to a linear gaussian SEM with a given
#' precision matrix and ordering on the nodes
#' 
#' @param siginv precision matrix
#' @param perm ordering of the nodes
#' @return B, the adjacency matrix
#' @return D, 
gauss_params <- function(siginv, perm){
  perm = as.numeric(perm)
  siginv = siginv[rev(perm), rev(perm)] # need to reverse ordering so lower triang since B upper triangular 
  L = ldl(siginv) # for some reason function puts L in lower part, and D on diagonal
  D = diag(diag(L))
  diag(L) = 1
  L = L[invPerm(rev(perm)), invPerm(rev(perm))] # reverse back to original ordering
  D = D[invPerm(rev(perm)), invPerm(rev(perm))] # reverse back to original ordering
  B = diag(nrow=length(perm)) - L
  return(list(B, D))
}

#' Get the precision matrix from the adjacency matrix
#' 
#' @param B The adjacency matrix
#' @param omega The noise matrix, by default identity
adj2prec <- function(B, omega=NULL) {
  p = nrow(B)
  id = diag(p)
  if (is.null(omega)) {
    return((id - B) %*% t(id - B)) 
  } else {
    return((id - B) %*% solve(omega) %*% t(id - B))
  }
}

########### Example ###########
# p = 3
# e = empty.graph(as.character(1:p))
# arc_set = matrix(c('1', '2', '2', '3'), ncol = 2, byrow = TRUE)
# arcs(e) = arc_set
# params = construct_B(e)
# init_data = gen_gaus_int_data(e, params, c(1, 3), c(50, 10))
# more_data = gen_gaus_int_data(e, params, c(1, 2), c(100, 100))
# lot_data = gen_gaus_int_data(e, params, c(1, 2, 3), c(10000, 10000, 10000))
# 
# # remaining DAGs in MEC
# e2 = empty.graph(as.character(1:p))
# arc_set = matrix(c('3', '2', '2', '1'), ncol = 2, byrow = TRUE)
# arcs(e2) = arc_set
# 
# e3 = empty.graph(as.character(1:p))
# arc_set = matrix(c('2', '3', '2', '1'), ncol = 2, byrow = TRUE)
# arcs(e3) = arc_set
# 
# unif_samps = cov_edge_sampler(e, MEC_unif_dist, 0, niters=1000) # sample uniformly from MEC, 0 is dummy since 
# e_nums = sum(sapply(1:1000, function(i) sum(abs(amat(unif_samps[[i]]) - amat(e))) == 0)) # should be ~333
# e2_nums = sum(sapply(1:1000, function(i) sum(abs(amat(unif_samps[[i]]) - amat(e2))) == 0)) # should be ~333
# e3_nums = sum(sapply(1:1000, function(i) sum(abs(amat(unif_samps[[i]]) - amat(e3))) == 0)) # should be ~333

