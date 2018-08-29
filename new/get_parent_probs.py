import argparse
from analysis import check_gies

parser = argparse.ArgumentParser(description='Calculate probabilities of each node being a parent of the target, given a dataset.')

parser.add_argument('--folder', type=str, help='Folder containing the dataset')
parser.add_argument('--target', type=int, help='Target node name')

args = parser.parse_args()

dag_folders = check_gies.get_dag_folders(args.folder)
check_gies.get_parent_probs_by_dag(dag_folders, args.target)


