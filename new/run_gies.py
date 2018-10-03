from utils import graph_utils
import numpy as np

folder = 'data/chain_test10/dags/dag0/entropy-dag-collection,n=1500,b=3,k=1/'
samples_folder = folder + 'samples'
temp_folder = folder + 'temp/'
obs_samples = np.loadtxt(samples_folder + '/intervention=-1.csv')
iv_samples = np.loadtxt(samples_folder + '/intervention=1.csv')
data = {-1: obs_samples, 1: iv_samples}

graph_utils._write_data(data, temp_folder + 'samples.csv', temp_folder + 'interventions.csv')
graph_utils.run_gies_boot(100, temp_folder + 'samples.csv', temp_folder + 'interventions.csv', temp_folder + 'sampled_dags/')
_, sampled_dags = graph_utils._load_dags(temp_folder + 'sampled_dags/', delete=False)
a = [d.arcs for d in sampled_dags]

