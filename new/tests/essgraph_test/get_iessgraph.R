library('bnlearn')
library('pcalg')

adj = read.table('tests/essgraph_test/tmp-graph.txt')
interventions = scan('tests/essgraph_test/tmp-interventions.txt')
b = empty.graph(as.character(1:ncol(adj)))

for (i in 1:nrow(adj)) {
    for (j in 1:ncol(adj)) {
        if (adj[i,j] == 1) {
            b = set.arc(b, as.character(i), as.character(j))
        }
    }
}
b_pc = as.graphNEL(b)

essgraph = dag2essgraph(b_pc, targets=interventions)
essgraph_adj = as(essgraph, 'matrix')
write.table(essgraph_adj, 'tests/essgraph_test/tmp-graph-r.txt', row.names=FALSE, col.names=FALSE)