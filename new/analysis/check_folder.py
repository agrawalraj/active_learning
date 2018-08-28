from config import DATA_FOLDER
import os

ASSUMED_FOLDERS = ['initial_dags', 'final_dags', 'samples']


def check_folder(folder):
    missing_folders = []

    dataset_folder = os.path.join(DATA_FOLDER, folder)
    dags_folder = os.path.join(dataset_folder, 'dags')
    for dag_folder in os.listdir(dags_folder):
        dag_folder = os.path.join(dags_folder, dag_folder)
        for strategy_folder in os.listdir(dag_folder):
            strategy_folder = os.path.join(dag_folder, strategy_folder)
            if os.path.isdir(strategy_folder):
                subfolders = set(os.listdir(strategy_folder))
                for assumed_folder in ASSUMED_FOLDERS:
                    if assumed_folder not in subfolders:
                        missing_folders.append(os.path.join(strategy_folder, assumed_folder))

    return sorted(missing_folders)
