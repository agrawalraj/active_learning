library(testthat)
library(bnlearn)
library(pcalg)
source('scoring.R')
source('sampling.r')

#' Format the way that we return interventions into the way expected by gies
#' 
#' @param intdata interventional data in the form of a list, with each element i
#' containing the data corresponding to the intervention at node i
#' @return list with $x being all samples, $targets being the intervention targets 
#' in order, and $target.index mapping each sample to the index in $targets of the
#' corresponding intervention
format_intdata = function(intdata) {
  data = matrix(nrow=0, ncol=10)
  target.index = c()
  targets = list()
  j = 1
  for (i in seq_along(intdata)) {
    d = intdata[[i]]
    if (!is.null(d)) {
      data = rbind(data, d)
      targets[[j]] = i
      target.index = c(target.index, rep(j, nrow(d)))
      j = j + 1
    }
  }
  return(list('x'=data, 'targets'=targets, 'target.index'=target.index))
}

test_that('interventional data is correctly generated', {
  p = 10
  K = 3
  gnodes = as.character(1:p)
  g = bnlearn::random.graph(gnodes, prob=1)
  B = construct_B(g)
  int_set = sample(1:p, K)
  int_data = gen_gaus_int_data(g, B, int_set, rep(1000, K))
  data = format_intdata(int_data)
  iessgraph = get_iessgraph(bnlearn::cpdag(g), g, as.character(int_set))
  score = new('GaussL0penIntScore', data$x, data$targets, data$target.index)
  gies.fit = gies(score)
  essgraph = gies.fit$repr
  # TODO: compare expected I-essential graph (iessgraph) to the one recovered from the data by GIES
  # and make sure they match (presumably, if data is being generated correctly, and there are enough
  # samples, they will exacly match)
})