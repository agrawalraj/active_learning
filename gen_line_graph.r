library(bnlearn)

all_line_graphs = function(p){
  all_dags = list()
  e = empty.graph(as.character(1:p))
  arc_vec = rep(as.character(1:p), each=2)
  arc_vec = arc_vec[2:(2 * p - 1)]
  arc_set = matrix(arc_vec, ncol = 2, byrow = TRUE)
  arcs(e) = arc_set
  all_dags[[1]] = e
  for(i in 1:(p-1)){
    e = reverse.arc(e, as.character(i), as.character(i + 1))
    all_dags[[i+1]] = e
  }
  return(all_dags)
}
