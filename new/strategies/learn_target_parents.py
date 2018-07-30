from utils import intervention_scores as scores


def learn_target_parents(g, target, data, config):
    dags = None
    scorer = scores.get_orient_parents_scorer(target, dags)

    interventions = []
    n_samples = []
    for k in range(config.max_interventions):
        intervention_scores = []
        for node in range(config.n_nodes):
            if node in interventions:
                intervention_scores.append(0)
            else:
                intervention_score = scorer(node)





