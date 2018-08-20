from __future__ import division  # in case python2 is used

import os
import shutil
import numpy as np
import config
import pandas as pd
import causaldag as cd


def bernoulli(p):
    return np.random.binomial(1, p)


def RAND_RANGE():
    return np.random.uniform(.25, 1) * (-1 if bernoulli(.5) else 1)


def run_min_imap(data_path, intervention_path, alpha=.05, gamma=1,
    n_iter=50000, save_step=100, path=config.TEMP_DAG_FOLDER, delete=False):
    # delete all DAGS in TEMP FOLDER
    if delete:
        try:
            shutil.rmtree(path)
            os.mkdir(path)
            print('All files deleted in ' + path)
        except Exception as e:
            os.mkdir(path)
            print('Made TEMP DAG directory')
    rfile = os.path.join(config.TOP_FOLDER, 'utils', 'minIMAP.r')
    r_command = 'Rscript {} {} {} {} {} {} {} {}'.format(rfile, data_path, intervention_path,
        str(alpha), str(gamma), str(n_iter), str(save_step), path)
    os.system(r_command)


def run_gies_boot(n_boot, data_path, intervention_path, path=config.TEMP_DAG_FOLDER, delete=False):
    # delete all DAGS in TEMP FOLDER
    if delete:
        try:
            shutil.rmtree(path)
            os.mkdir(path)
            print('All files deleted in ' + path)
        except Exception as e:
            os.mkdir(path)
            print('Made TEMP DAG directory')
    rfile = os.path.join(config.TOP_FOLDER, 'utils', 'run_gies.r')
    r_command = 'Rscript {} {} {} {} {}'.format(rfile, n_boot, data_path, intervention_path, path)
    os.system(r_command)


def _write_data(data):
    """
    Helper function to write interventional data to files so that it can be used by R
    """
    # clear current data
    open(config.TEMP_SAMPLES_PATH, 'w').close()
    open(config.TEMP_INTERVENTIONS_PATH, 'w').close()

    iv_nodes = []
    for iv_node, samples in data.items():
        with open(config.TEMP_SAMPLES_PATH, 'ab') as f:
            np.savetxt(f, samples)
        iv_nodes.extend([iv_node+1 if iv_node != -1 else -1]*len(samples))
    pd.Series(iv_nodes).to_csv(config.TEMP_INTERVENTIONS_PATH, index=False)


def _load_dags():
    """
    Helper function to load the DAGs generated in R
    """
    adj_mats = []
    paths = os.listdir(config.TEMP_DAG_FOLDER)
    for file_path in paths:
        if 'score' not in file_path and '.DS_Store' not in file_path:
            adj_mat = pd.read_csv(os.path.join(config.TEMP_DAG_FOLDER, file_path))
            adj_mats.append(adj_mat.as_matrix())
    return [cd.DAG.from_amat(adj) for adj in adj_mats]

def probability_shrinkage(prob):
    return 2 * min(1 - prob, prob)

