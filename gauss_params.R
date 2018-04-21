

gauss_params <- function(J, perm){
  J = J[perm, perm]
  L = t(chol(J))
  L = L/diag(L)
  B = diag(nrow=length(perm)) - L
  return(B)
}

