library('bnlearn')
library('pcalg')

b = empty.graph(as.character(0:4))

b = set.arc(b, '0', '3')
b = set.arc(b, '1', '3')
b = set.arc(b, '2', '3')
b = set.arc(b, '2', '4')
b = set.arc(b, '3', '4')

# should have 0 undirected edges
essgraph = cpdag(b, moral=TRUE)
essgraph2 = cpdag(b, moral=FALSE)
essgraph_pc = dag2cpdag(as.graphNEL(b))
print(essgraph)
print(essgraph2)
print(essgraph_pc)
# moral=FALSE gives correct result