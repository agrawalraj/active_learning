import argparse
from config import DATA_FOLDER
import os
from strategies.simulator import GenerationConfig

parser = argparse.ArgumentParser(description='Create a bunch of DAGs.')

parser.add_argument('--variables', '-p', type=int, help='number of nodes in each DAG')
parser.add_argument('--sparsity', '-s', type=float, help='sparsity of each DAG')
parser.add_argument('--dags', '-d', type=int, help='number of DAGs')

parser.add_argument('--folder', type=str, help='Folder in which to save the DAGs')

args = parser.parse_args()

G_CONFIG = GenerationConfig(
    n_nodes=args.variables,
    edge_prob=args.sparsity,
    n_dags=args.dags
)
gdags = G_CONFIG.save_dags(os.path.join(DATA_FOLDER, args.folder))


