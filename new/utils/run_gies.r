library(pcalg)
library(gRbase)

source('./utils/gies_helper.r') # ASSUMES RUN FROM NEW DIRECTORY

args = commandArgs(trailingOnly=TRUE)
n_boot = as.numeric(args[1])
data_path = args[2]
intervention_path = args[3]
path = args[4]

data = as.data.frame(read.table(data_path))
interventions = as.character(read.csv(intervention_path)[, 1])

boot_samps = bootstrap_gies(n_boot, data, interventions, path)
