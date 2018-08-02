library(bnlearn)
library(mvtnorm)
library(pcalg)
library(gRbase)

incident_mat = function(g){
  arcset = matrix(as.numeric(arcs(g)), ncol=2)
  p = nnodes(g)
  incident = matrix(0, p, p)
  incident[arcset] = 1
  return(incident)
}

ci_min_imap_fz = function(corr_mat, n, perm, alpha){
  # G_star - bnlearn object 
  # perm - character vector of nodes
  
  # returns bnlearn object of minimal i-map
  
  p = length(perm) # number of nodes
  e = empty.graph(as.character(1:p))
  arc_vec = c()
  perm = as.character(perm)
  for(i in 1:(p-1)){
    pi_i = perm[i]
    for(j in (i + 1):p){
      pi_j = perm[j]
      S = perm[1:(j-1)]
      S = S[S != pi_i]
      cutoff = qnorm(1 - alpha/2)
      fz_stat = condIndFisherZ(as.numeric(pi_i), as.numeric(pi_j), as.numeric(S), corr_mat, n, cutoff)
      if(fz_stat == FALSE){
        arc_vec = c(arc_vec, pi_i, pi_j)
      }
      
    }
  }
  if (length(arc_vec) > 0){
    arc_set = matrix(arc_vec, ncol=2, byrow=TRUE,
                     dimnames=list(NULL, c("from", "to")))
    arcs(e) = arc_set
  }
  return(e)
}

update_emp_DAG = function(g_hat, new_perm, prev_perm, j, corr_mat, n, alpha_level){
  p = nnodes(g_hat)
  e = empty.graph(as.character(1:p))
  inc_mat = incident_mat(g_hat)
  pi_j = new_perm[j]
  pi_jp1 = new_perm[j+1]
  S_j = new_perm[1:(j-1)]
  S_jp1 = new_perm[1:j]
  cutoff = qnorm(1 - alpha_level/2)
  if(j > 1){
    for (i in 1:(j-1)){ # update edges before swap
      pi_i = new_perm[i] # = prev_perm[i]
      inc_mat[[as.numeric(pi_i), as.numeric(pi_j)]] = 0
      inc_mat[[as.numeric(pi_i), as.numeric(pi_jp1)]] = 0
      S_i = S_j[S_j != pi_i] 
      S_ip1 = S_jp1[S_jp1 != pi_i]
      fz_stat = condIndFisherZ(as.numeric(pi_i), as.numeric(pi_j), as.numeric(S_i), corr_mat, n, cutoff)
      if(fz_stat == FALSE){
        inc_mat[[as.numeric(pi_i), as.numeric(pi_j)]] = 1
      }
      fz_statp1 = condIndFisherZ(as.numeric(pi_i), as.numeric(pi_jp1), as.numeric(S_ip1), corr_mat, n, cutoff)
      if(fz_statp1 == FALSE){
        inc_mat[[as.numeric(pi_i), as.numeric(pi_jp1)]] = 1
      }
    }}
  # reverse arrow if (j, j+1) had an edge
  ind_arrow = inc_mat[as.numeric(prev_perm[j]), as.numeric(prev_perm[j+1])]
  if(ind_arrow == 1){
    inc_mat[[as.numeric(prev_perm[j]), as.numeric(prev_perm[j+1])]] = 0
    inc_mat[[as.numeric(pi_j), as.numeric(pi_jp1)]] = 1
  }
  arc_set = which(inc_mat == 1, arr.ind=TRUE)
  arc_set = matrix(apply(arc_set, 2, as.character), ncol=2)
  colnames(arc_set) = c("from", "to")
  arcs(e) = arc_set
  return(e)
}

calc_intervention_prob = function(g, data, interventions){
  data = as.data.frame(data)
  node_scores = bnlearn::score(g, data, type='bge', by.node=TRUE)
  unique_interventions = unique(interventions)
  for (i in 1:length(unique_interventions)){
    intervened_node = unique_interventions[i]
    parents_intervened_node = bnlearn::parents(g, intervened_node)
    g_sub = empty.graph(c(intervened_node, parents_intervened_node))
    if(length(parents_intervened_node) > 0){
      for(j in 1:length(parents_intervened_node)){
        g_sub = set.arc(g_sub, parents_intervened_node[j],intervened_node)
      }
    }
    mask = interventions != intervened_node
    data_sub = data[c(intervened_node, parents_intervened_node)]
    colnames_sub = colnames(data_sub)
    data_sub = data_sub[mask, ]
    data_sub = as.data.frame(data_sub) # R does does weird stuff converting data frame to vector
    colnames(data_sub) = colnames_sub
    intervened_node_score = bnlearn::score(g_sub, data_sub, type='bge', by.node=TRUE)
    intervened_node_score = intervened_node_score[intervened_node] # Just want score of intervened node
    node_scores[as.numeric(intervened_node)] = intervened_node_score
  }
  return(sum(node_scores))
}
 
p_guas_fz2 = function(data, interventions, corr_mat, n, pi, alpha, gamma){
  pi_imap_hat = ci_min_imap_fz(corr_mat, n, pi, alpha)
  data_dep_prior = -gamma * narcs(pi_imap_hat)
  p_lik = calc_intervention_prob(pi_imap_hat, data, interventions)
  return(list(p_lik + data_dep_prior, pi_imap_hat))
}

p_guas_fz_cached = function(g_prev, pi_new, pi_prev, j, data, interventions, corr_mat, n, alpha, gamma){
  pi_imap_hat = update_emp_DAG(g_prev,pi_new,pi_prev,j, corr_mat, n, alpha)
  data_dep_prior = -gamma * narcs(pi_imap_hat)
  p_lik = calc_intervention_prob(pi_imap_hat, data, interventions)
  return(list(p_lik + data_dep_prior, pi_imap_hat))
}

random_transposition = function(perm){
  p = length(perm)
  root_idx = sample(1:p, 1)
  left_right = sample(c(-1, 1), 1)
  neighb_idx = root_idx + left_right
  if (neighb_idx == 0){ # root_idx = 1, left_right = -1
    neighb_idx = p
  } else if (neighb_idx == p + 1){
    neighb_idx = 1
  }
  elem1 = perm[root_idx]
  elem2 = perm[neighb_idx]
  perm_next = perm
  perm_next[root_idx] = elem2 # transposition
  perm_next[neighb_idx] = elem1
  return(list(perm_next, min(root_idx, neighb_idx)))
}

minIMAP_MCMC = function(data_path, intervention_path, alpha=.05, gamma=1, n_iter=500, save_step=100, path='../data/TEMP_DAGS/'){
  data = as.data.frame(read.csv(data_path))
  p = ncol(data)
  colnames(data) = as.character(1:p)
  interventions = as.character(read.csv(intervention_path)[, 1])
  corr_mat = cor(data[interventions == -1, ]) # -1 is flag for observational data
  corr_mat = cor(data)
  all_targets = list()
  all_targets[[1]] = integer(0) # observation data marker
  possible_interventions = unique(interventions)
  if(length(possible_interventions) > 0){
    for(i in 1:length(possible_interventions)){
      if(as.numeric(possible_interventions[i]) != -1){
        all_targets[[i + 1]] = as.numeric(possible_interventions[i])
      }
    }
  }
  gie_score_fn <- new("GaussL0penIntScore", data, all_targets, as.numeric(interventions)) # BIC score
  gies.fit <- gies(gie_score_fn)
  weights = gies.fit$repr$weight.mat()
  weights[weights != 0, ] = 1 # convert to adjacency matrix
  pi_0 = as.character(topoSort(weights))
  n = dim(data)[1]
  p = length(pi_0)
  pi_prev = pi_0
  accepted = 0
  prev = p_guas_fz2(data, interventions, corr_mat, n, pi_prev, alpha, gamma) # log scale
  p_prev = prev[[1]]
  emp_dag_prev = prev[[2]]
  scores = c()
  emp_dags = list()
  for (i in 1:n_iter){
    out = random_transposition(pi_prev)
    x_prop = out[[1]]
    j = out[[2]]
    if(x_prop[p] != pi_prev[1]){ # fast update only works if transposition doesn't occur at end
      curr = p_guas_fz_cached(emp_dag_prev, x_prop, pi_prev, j, data, interventions, corr_mat, n, alpha, gamma) # log scale
    } else{
      curr = p_guas_fz2(data, interventions, corr_mat, n, x_prop, alpha, gamma)
    }
    p_prop = curr[[1]]
    emp_dag_curr = curr[[2]]
    pdfratio = exp(p_prop - p_prev) # q_prop(pi_prev) = q_prop(pi_next)
    if (runif(1) < min(c(1, pdfratio))){
      pi_prev = x_prop
      accepted = accepted + 1
      p_prev = p_prop
      emp_dag_prev = emp_dag_curr
    }
    emp_dags[[i]] = emp_dag_prev
    scores = c(scores, p_prev)
    print(i)
    print('what')
    print(save_step)
    if(i %% save_step == 0){
      print('about data')
      index = i / save_step
      write.csv(amat(emp_dag_prev), paste(path, index, sep=''), row.names=FALSE)
      print('saved data')
    }
  }
  write.csv(scores, paste(path, 'scores', sep=''), row.names=FALSE)
  return(list(scores, accepted, emp_dags))
}

args = commandArgs(trailingOnly=TRUE)
data_path = args[1]
intervention_path = args[2]
alpha = as.numeric(args[3])
gamma = as.numeric(args[4])
n_iter = as.numeric(args[5])
save_step = as.numeric(args[6])
path = args[7]
print(n_iter)
print(class(save_step))

samps = minIMAP_MCMC(data_path, intervention_path, alpha, gamma, n_iter, save_step, path)
