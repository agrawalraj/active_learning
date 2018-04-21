library(pcalg);
library(bnlearn);
library(graph);

score.intervention.dag = function(essgraph, true_dag, intervened_nodes, verbose=FALSE) {
  # calculates the number of edges learned by cut set and Meek rules
  # if the true DAG is true_dag and infinite-sample interventions
  # are performed at intervened_nodes

  graph.after.cut = essgraph
  for (node in intervened_nodes) {
    for (child in bnlearn::children(true_dag, node)) {
      graph.after.cut = bnlearn::set.arc(graph.after.cut, node, child)
    }
    for (parent in bnlearn::parents(true_dag, node)) {
      graph.after.cut = bnlearn::set.arc(graph.after.cut, parent, node)
    }
  }

  updated = pcalg::addBgKnowledge(as.graphNEL(graph.after.cut), verbose=verbose, checkInput = FALSE)
  updated = bnlearn::as.bn(updated) 
  # because essgraph double counts undirected edges
  score = nrow(bnlearn::directed.arcs(updated)) - nrow(bnlearn::directed.arcs(essgraph))
  return(list('updated' = updated, 'score' = score))
}

score.intervention.dags = function(essgraph, dags, intervened_nodes) {
  # calculates the Monte-Carlo expected value of the number of edges
  # learned by cut set and Meek rules, with dags[i] having current
  # probability ps[i]
  s <- 0
  updated <- list()
  for (dag in dags) {
    intervention.result = score.intervention.dag(essgraph, dag, intervened_nodes)
    updated = append(updated, list(intervention.result$updated))
    s = s + intervention.result$score
  }
  return(list('score' = s / length(dags), 'updated' = updated))
}

essgraph = bnlearn::empty.graph(nodes = as.character(1:3))
essgraph = bnlearn::set.edge(essgraph, '1', '2')
essgraph = bnlearn::set.edge(essgraph, '1', '3')
essgraph = bnlearn::set.arc(essgraph, '2', '3')

g1 = bnlearn::empty.graph(nodes = as.character(1:3))
g1 = bnlearn::set.arc(g1, '1', '2')
g1 = bnlearn::set.arc(g1, '1', '3')
g1 = bnlearn::set.arc(g1, '2', '3')

g2 = bnlearn::empty.graph(nodes = as.character(1:3))
g2 = bnlearn::set.arc(g2, '2', '1')
g2 = bnlearn::set.arc(g2, '1', '3')
g2 = bnlearn::set.arc(g2, '2', '3')

intervention.result1 = score.intervention.dag(essgraph, g1, c('2'))
intervention.result2 = score.intervention.dag(essgraph, g2, c('2'))
gs = list(g1, g2)
intervention.results = score.intervention.dags(essgraph, gs, c('2'))

