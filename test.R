library(bnlearn)
source('sampling.R')

p = 3
e = empty.graph(as.character(1:p))
arc_set = matrix(c('1', '2', '3', '2'), ncol = 2, byrow = TRUE)
arcs(e) = arc_set

# e2 = empty.graph(as.character(1:p))
# arc_set = matrix(c('2', '1', '2', '3'), ncol = 2, byrow = TRUE)
# arcs(e2) = arc_set
a = covered_edges(e2)