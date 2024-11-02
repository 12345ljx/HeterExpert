from collections import defaultdict, Counter
import numpy as np
import torch
from transformers import set_seed
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.preprocessing import normalize
from k_means_constrained import KMeansConstrained

import sys
sys.path.append('/usr/workdir/HeterExpert/models')
from importance_llama import LlamaForCausalLM

random_seed = 42
set_seed(random_seed)

model_path = '/usr/workdir/models/llama-3.2-1B'
model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)

data_path = '/usr/workdir/HeterExpert/Neuron_Importance/score5000'
domains_data = np.load(f'{data_path}/importance_score.npy')  # [num_layers, num_neurons, num_domains]
num_layers, num_neurons, num_domains = domains_data.shape
num_modules = 512
assert num_neurons % num_modules == 0
size_module = num_neurons // num_modules

domains_data_reduced = np.zeros((num_layers, num_modules, num_domains))

num_layers = model.config.num_hidden_layers

for layer in range(num_layers):
    params = model.model.layers[layer].mlp.gate_proj.weight.data  # [dff_hidden_size, hidden_size]
    # params += model.model.layers[layer].mlp.up_proj.weight.data   # TODO
    params_norm = normalize(params, axis=1)
    
    # clusters = fcluster(linkage(params_norm, method='ward'), num_modules, criterion='maxclust')  # idx range(1, num_modules+1)
    # clusters = KMeans(n_clusters=num_modules, random_state=random_seed).fit_predict(params_norm)  # idx range(0, num_modules)
    clusters = KMeansConstrained(n_clusters=num_modules, size_min=size_module, size_max=size_module, random_state=random_seed).fit_predict(params_norm)  # idx range(0, num_modules)
    
    start_idx = 0  # 1 for fcluster
    clusters_idx_range = range(start_idx, num_modules + start_idx)
    
    experts_size = Counter([np.sum(clusters == i) for i in clusters_idx_range])
    assert sum(experts_size.values()) == num_modules
    experts_size_sorted = sorted(experts_size.items(), key=lambda x: x[0])
    print(f'Layer {layer}: {experts_size_sorted}')

    module2neurons = np.zeros((num_modules, size_module))
    for cluster_idx in clusters_idx_range:
        neurons_idx = np.where(clusters == cluster_idx)[0]
        module2neurons[cluster_idx] = neurons_idx
        cluster_scores = np.sum(domains_data[layer][neurons_idx], axis=0)
        domains_data_reduced[layer][cluster_idx - start_idx] = cluster_scores
    
# np.save(f'{data_path}/importance_score_reduced.npy', domains_data_reduced)
np.save(f'{data_path}/importance_score_reduced.npz', domains_data_reduced=domains_data_reduced, module2neurons=module2neurons)