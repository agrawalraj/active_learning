source('sampling.r')
library(bnlearn)


# ========
# Test that gauss_params gives back original edge weights
# ========

random_up_tri = function(p, edge_prob) {
  g = bnlearn::random.graph(as.character(1:p), prob=edge_prob)
  perm = bnlearn::node.ordering(g)
  nodes(g) = perm
  return(g)
}
p = 3
id = diag(p)
g = random_up_tri(p, .5)

# Ba = matrix(c(0,0,2,0,0,3,0,0,0), nrow=3, ncol=3, byrow=TRUE)
# Sa = (id - Ba) %*% t(id - Ba)
# r = reversible.arcs(g)
perm = as.numeric(node.ordering(g))
B = construct_B(g)
siginv = (id - B) %*% t(id - B)

# B2 = B[rev(perm), rev(perm)]
# siginv2 = (id - B2) %*% t(id - B2)
# siginv3 = siginv[rev(perm), rev(perm)]
# compute-then-reorder same as reorder-then-compute
# print(all.equal(siginv2, siginv3))

g_params = gauss_params(siginv, perm)
B_rec = g_params[[1]]
omega_rec = g_params[[2]]
siginv_rec = (id - B_rec) %*% solve(omega_rec) %*% t(id - B_rec)
print(all.equal(B, B_rec))
print(all.equal(siginv, siginv_rec))

p2 = 4
g2 = bnlearn::random.graph(as.character(1:p2), prob=.5)
B2 = construct_B(g2)
siginv2 = (diag(p2) - B2) %*% t(diag(p2) - B2)
perm2 = as.numeric(node.ordering(g2))
g_params = gauss_params(siginv2, perm2)
B2_rec = g_params[[1]]
omega2_rec = g_params[[2]]
siginv2_rec = (diag(p2) - B2_rec) %*% solve(omega2_rec) %*% t(diag(p2) - B2_rec)
print(all.equal(B2, B2_rec))
print(all.equal(siginv2, siginv2_rec))