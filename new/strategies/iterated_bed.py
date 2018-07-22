from utils import intervention_scores as scores
import numpy as np
from utils import sampler
from utils import graph_utils


# TODO: check when bed_score and sample_dags finished
def iterated_bed(g, siginv, data, config):
    interventions = []
    n_samples = []
    samp_dags = sampler.sample_dags(g, siginv, data)
    essgraph = graph_utils.get_essgraph(g)
    for k in range(config.max_interventions):
        intervention_scores = []
        for node in range(config.n_nodes):
            if node in interventions:
                intervention_scores.append(0)
            else:
                intervention_score = scores.bed_score(node, samp_dags, essgraph)
                intervention_scores.append(intervention_score)
        best_intervention = np.argmax(intervention_scores)
        interventions.append(best_intervention)
        n = config.n_samples / (config.n_batches * config.max_interventions)
        if int(n) != n:
            raise ValueError('number of samples divided by (number of batches * max number of interventions) is not an integer')
        n_samples.append(int(n))

    return interventions, n_samples

