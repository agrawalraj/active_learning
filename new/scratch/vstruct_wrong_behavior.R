library('bnlearn')

b = empty.graph(as.character(1:4))

b = set.arc(b, '1', '2')
b = set.arc(b, '1', '4')
b = set.arc(b, '2', '3')
b = set.arc(b, '2', '4')
b = set.arc(b, '3', '4')

v = vstructs(b, arcs=TRUE, moral=FALSE)
v2 = vstructs(b, arcs=TRUE)
print(v)
print(v2)
