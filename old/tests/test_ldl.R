library(testthat)
library(bnlearn)
source('sampling.r')

test_that("same number of edges", {
  p = 5
  gnodes = as.character(1:p)
  g = bnlearn::random.graph(gnodes, 1, prob=.5)
  B = construct_B(g)
  siginv = adj2prec(B)
  new_perm = sample(gnodes, size=p)
  gparams = gauss_params(siginv, new_perm)
  B2 = gparams[[1]]
  print(sum(B!=0))
  print(sum(B2!=0))
})

test_that("Adjacency matrix recovered correctly for complete graph", {
  p = 5
  gnodes = as.character(1:p)
  g = bnlearn::random.graph(gnodes, 1, prob=1)
  B = construct_B(g)
  siginv = adj2prec(B)
  new_perm = sample(gnodes, size=p)
  gparams = gauss_params(siginv, new_perm)
  B2 = gparams[[1]]
  omega2 = gparams[[2]]
  siginv2 = adj2prec(B2, solve(omega2))
  expect_equal(siginv, siginv2)
})

test_that("Adjacency matrix recovered correctly for line graph", {
  p = 5
  gnodes = as.character(1:p)
  g = bnlearn::empty.graph(gnodes)
  for (i in 2:p) {
    g = bnlearn::set.arc(g, as.character(i-1), as.character(i))
  }
  B = construct_B(g)
  siginv = adj2prec(B)
  new_source = sample(2:(p-1), 1)
  new_perm = c(new_source, (new_source-1):1, (new_source+1):p)
  new_perm = as.character(new_perm)
  gparams = gauss_params(siginv, new_perm)
  B2 = gparams[[1]]
  omega2 = gparams[[2]]
  siginv2 = adj2prec(B2, solve(omega2))
  expect_equal(siginv, siginv2)
})

test_that("Adjacency matrix recovered correctly for random graph", {
  p = 5
  gnodes = as.character(1:p)
  g = bnlearn::random.graph(gnodes, prob=.5)
  B = construct_B(g)
  siginv = adj2prec(B)
  g2 = rand_from_MEC(g)
  new_perm = node.ordering(g2)
  gparams = gauss_params(siginv, new_perm)
  B2 = gparams[[1]]
  omega2 = gparams[[2]]
  siginv2 = adj2prec(B2, solve(omega2))
  expect_equal(siginv, siginv2)
})