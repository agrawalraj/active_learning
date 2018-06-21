library('bnlearn')

adj = read.table(paste('tests/essgraph_test/tmp-graph.txt', sep=''))
b = empty.graph(as.character(1:ncol(adj)))

for (i in 1:nrow(adj)) {
    for (j in 1:ncol(adj)) {
        if (adj[i,j] == 1) {
            b = set.arc(b, as.character(i), as.character(j))
        }
    }
}

essgraph = cpdag(b)
essgraph_adj = amat(essgraph)
write.table(essgraph_adj, 'tests/essgraph_test/tmp-graph-r.txt', row.names=FALSE, col.names=FALSE)