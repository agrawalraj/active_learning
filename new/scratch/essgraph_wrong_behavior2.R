library('bnlearn')

b = empty.graph(as.character(0:4))

b = set.arc(b, '0', '1')
b = set.arc(b, '0', '3')
b = set.arc(b, '1', '3')
b = set.arc(b, '1', '4')
b = set.arc(b, '2', '4')

# should have 3 undirected edges
essgraph = cpdag(b, moral=TRUE)
essgraph2 = cpdag(b, moral=FALSE)
print(essgraph)
print(essgraph2)
# moral=TRUE gives correct result