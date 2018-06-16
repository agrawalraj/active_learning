library(pcalg)
library(bnlearn)
library(graph)

#' Given the true DAG and its essential graph, find the I-essential graph after an intervention I
#' 
#' @param essgraph : the essential graph of the true DAG (bnlearn graph)
#' @param true_dag : the true DAG (bnlearn graph)
#' @param intervened_nodes
#' @return the I-essential graph (bnlearn graph)
get_iessgraph = function(essgraph, true_dag, intervened_nodes, verbose=FALSE) {
  graph.after.cut = essgraph
  for (node in intervened_nodes) {
    for (child in bnlearn::children(true_dag, node)) {
      graph.after.cut = bnlearn::set.arc(graph.after.cut, node, child)
    }
    for (parent in bnlearn::parents(true_dag, node)) {
      graph.after.cut = bnlearn::set.arc(graph.after.cut, parent, node)
    }
  }
  iessgraph = pcalg::addBgKnowledge(as.graphNEL(graph.after.cut), verbose=verbose, checkInput = FALSE)
  iessgraph = bnlearn::as.bn(iessgraph) 
  return(iessgraph)
}

#' Calculate the number of edges learned by an intervention
#' 
#' @param essgraph : the essential graph of the true DAG (bnlearn graph)
#' @param true_dag : the true DAG (bnlearn graph)
#' @param intervened_nodes
#' @return a list containing the updated graph and the number of newly oriented edges
score.intervention.dag = function(essgraph, true_dag, intervened_nodes, verbose=FALSE) {
  iessgraph = get_iessgraph(essgraph, true_dag, intervened_nodes, verbose=verbose)
  score = nrow(bnlearn::directed.arcs(iessgraph)) - nrow(bnlearn::directed.arcs(essgraph))
  return(list('updated' = iessgraph, 'score' = score))
}

#' Calculates the average number of edges learned by an intervention for a set of DAGs
#' @param essgraph : the essential graph of the true DAG (bnlearn graph)
#' @param dags
#' @param intervened_nodes
score.intervention.dags = function(essgraph, dags, intervened_nodes) {
  s = 0
  updated = list()
  for (dag in dags) {
    intervention.result = score.intervention.dag(essgraph, dag, intervened_nodes)
    updated = append(updated, list(intervention.result$updated))
    s = s + intervention.result$score
  }
  return(list('score' = s / length(dags), 'updated' = updated))
}

# essgraph = bnlearn::empty.graph(nodes = as.character(1:3))
# essgraph = bnlearn::set.edge(essgraph, '1', '2')
# essgraph = bnlearn::set.edge(essgraph, '1', '3')
# essgraph = bnlearn::set.arc(essgraph, '2', '3')
# 
# g1 = bnlearn::empty.graph(nodes = as.character(1:3))
# g1 = bnlearn::set.arc(g1, '1', '2')
# g1 = bnlearn::set.arc(g1, '1', '3')
# g1 = bnlearn::set.arc(g1, '2', '3')
# 
# g2 = bnlearn::empty.graph(nodes = as.character(1:3))
# g2 = bnlearn::set.arc(g2, '2', '1')
# g2 = bnlearn::set.arc(g2, '1', '3')
# g2 = bnlearn::set.arc(g2, '2', '3')
# 
# intervention.result1 = score.intervention.dag(essgraph, g1, c('2'))
# intervention.result2 = score.intervention.dag(essgraph, g2, c('2'))
# gs = list(g1, g2)
# intervention.results = score.intervention.dags(essgraph, gs, c('2'))

