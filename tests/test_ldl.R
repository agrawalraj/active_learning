library(testthat)
library(bnlearn)
source('sampling.r')

test_that("Test gauss_params complete graph", {
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

test_that("Test gauss_params line graph", {
  p = 5
  gnodes = as.character(1:p)
  g = bnlearn::empty.graph(gnodes)
  for (i in 2:p) {
    g = bnlearn::set.arc(g, as.character(i-1), as.character(i))
  }
  B = construct_B(g)
  siginv = adj2prec(B)
  new_source = sample(1:p, 1)
  new_perm = c(new_source, (new_source-1):1, (new_source+1):p)
  new_perm = as.character(new_perm)
  gparams = gauss_params(siginv, new_perm)
  B2 = gparams[[1]]
  omega2 = gparams[[2]]
  siginv2 = adj2prec(B2, solve(omega2))
  expect_equal(siginv, siginv2)
})