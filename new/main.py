
# coding: utf-8

# In[ ]:


from strategies import random_nodes, learn_target_parents, edge_prob
import numpy as np
import os
from config import DATA_FOLDER
from strategies.simulator import GenerationConfig, SimulationConfig, simulate
from analysis.check_gies import check_gies
np.random.seed(1729)


# In[ ]:


N_NODES = 50
DAG_FOLDER = os.path.join(DATA_FOLDER, 'medium')
STRATEGIES = {
    'random': random_nodes.random_strategy,
    'learn-parents': learn_target_parents.create_learn_target_parents(N_NODES-3, 100),
    'edge-prob': edge_prob.create_edge_prob_strategy(N_NODES-3, 100)
}


# ### Generate and save DAGs

# In[ ]:


G_CONFIG = GenerationConfig(
    n_nodes=N_NODES,
    edge_prob=.5,
    n_dags=3
)
gdags = G_CONFIG.save_dags(DAG_FOLDER)


# ### Run random strategy on each DAG

# In[ ]:


SIM_CONFIG_RANDOM = SimulationConfig(
    starting_samples=250,
    n_samples=60,
    n_batches=2,
    max_interventions=2,
    strategy='random',
    intervention_strength=2,
)


# In[ ]:


for i, gdag in enumerate(gdags):
    print('=== Simulating strategy for DAG %d' % (i))
    simulate(STRATEGIES['random'], SIM_CONFIG_RANDOM, gdag, os.path.join(DAG_FOLDER, 'dag%d' % i, 'random'))


# ### Run edge-prob strategy on each DAG

# In[ ]:


SIM_CONFIG_EDGE_PROB = SimulationConfig(
    starting_samples=250,
    n_samples=60,
    n_batches=2,
    max_interventions=2,
    strategy='edge-prob',
    intervention_strength=2,
)


# In[ ]:


for i, gdag in enumerate(gdags):
    print('=== Simulating strategy for DAG %d' % (i))
    simulate(STRATEGIES['edge-prob'], SIM_CONFIG_EDGE_PROB, gdag, os.path.join(DAG_FOLDER, 'dag%d' % i, 'edge-prob'))


# ### Run learn-parents strategy on each DAG

# In[ ]:


SIM_CONFIG_PARENTS = SimulationConfig(
    starting_samples=250,
    n_samples=60,
    n_batches=2,
    max_interventions=2,
    strategy='learn-parents',
    intervention_strength=2,
)


# In[ ]:


for i, gdag in enumerate(gdags):
    print('=== Simulating strategy for DAG %d' % (i))
    simulate(STRATEGIES['learn-parents'], SIM_CONFIG_EDGE_PROB, gdag, os.path.join(DAG_FOLDER, 'dag%d' % i, 'learn-parents'))


# ### Check gathered data

# In[ ]:


random_rhs = []
for d, gdag in enumerate(gdags):
    folder = os.path.join(DAG_FOLDER, 'dag%d' % d)
    parent_probs, rh = check_gies(folder, 'random', N_NODES-3)
    random_rhs.append(rh)


# In[ ]:


random_tprs = [rh.tpr for rh in random_rhs]
random_fprs = [rh.fpr for rh in random_rhs]
print(random_tprs)
print(random_fprs)


# In[ ]:


edge_prob_rhs = []
for d, gdag in enumerate(gdags):
    folder = os.path.join(DAG_FOLDER, 'dag%d' % d)
    parent_probs, rh = check_gies(folder, 'edge-prob', N_NODES-3)
    edge_prob_rhs.append(rh)


# In[ ]:


edge_prob_tprs = [rh.tpr for rh in edge_prob_rhs]
edge_prob_fprs = [rh.fpr for rh in edge_prob_rhs]
print(edge_prob_tprs)
print(edge_prob_fprs)


# In[ ]:


learn_parents_rhs = []
for d, gdag in enumerate(gdags):
    folder = os.path.join(DAG_FOLDER, 'dag%d' % d)
    parent_probs, rh = check_gies(folder, 'learn-parents', N_NODES-3)
    learn_parents_rhs.append(rh)


# In[ ]:


learn_parents_tprs = [rh.tpr for rh in learn_parents_rhs]
learn_parents_fprs = [rh.fpr for rh in learn_parents_rhs]
print(learn_parents_tprs)
print(learn_parents_fprs)

