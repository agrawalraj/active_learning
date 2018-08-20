
import graph_utils
import causaldag as cd
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score

def parent_probs(adj_mats):
	dags = [cd.from_amat(adj) for adj in adj_mats]
	parent_counts = defaultdict(int)
	node_set = dags[0].nodes
	for node in node_set:
	    parent_counts[node] = 0
	for dag in dags:
	    for p in dag.parents[target]:
	        parent_counts[p] += 1
	parent_probs = {p: c/len(dags) for p, c in parent_counts.items()}
	return parent_probs


def plot_roc(target, true_adj_mat, adj_mats):
	avg_inc_mat = np.zeros(true_adj_mat.shape)
	for adj_mat in adj_mats:
		avg_inc_mat += graph_utils.adj2inc(adj_mat)
	true_inc_mat = graph_utils.adj2inc(true_adj_mat)
	roc_curve(true_inc_mat[:, target], avg_inc_mat[:, target])

if __name__ == '__main__':
	adj_true = np.loadtxt('./data/dataset_5000/graph_2/adjacency.csv')
	g = cd.from_amat(adj_true)
	dict_weights = {}
	for arc in g.arcs:
		dict_weights[arc] = adj_true[arc[0], arc[1]]
	gdag = cd.GaussDAG(nodes=list(range(50)), arcs=dict_weights)
	all_data = [gdag.sample(250)]
	interventions = [-1] * 250
	all_iv = np.random.randint(0, 50, 10)
	for iv in all_iv:
		interventions += [iv] * 25
		g_iv = cd.GaussIntervention(mean=2, variance=1)
		all_data.append(gdag.sample_interventional({iv: g_iv}, 25))

	all_data = np.vstack(all_data)
	interventions = np.array(interventions)
	interventions[interventions != -1] = interventions[interventions != -1] + 1
	np.savetxt('./random_data', all_data)
	np.savetxt('./random_interventions', interventions)
	interventions[interventions != -1] = interventions[interventions != -1] - 1
	graph_utils.run_gies_boot(200, './random_data', './random_interventions')
	adj_mats = graph_utils.load_adj_mats()
	np.save('./data/dataset_5000/graph_2/adj_mats_random', adj_mats)

