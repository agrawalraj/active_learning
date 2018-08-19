import os


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def make_data(num_nodes, sample_path):
    possible_interventions = range(-1, num_nodes)
    interventions = []
    all_data = []
    for iv in possible_interventions:
        path = sample_path + '/intervention=' + str(iv) + '.csv'
        data = np.loadtxt(path)
        if data.shape[0] > 0:
            interventions += [iv] * data.shape[0]
            all_data.append(data)
    all_data = np.vstack(all_data)
    interventions = np.array(interventions)
    return all_data, interventions