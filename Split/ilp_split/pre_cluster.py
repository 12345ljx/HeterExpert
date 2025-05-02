import os
from collections import defaultdict, Counter
import numpy as np
import torch
from transformers import set_seed, AutoModelForCausalLM
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.preprocessing import normalize
from k_means_constrained import KMeansConstrained

# NUM_MODULES = 512
NUM_MODULES = 128

def get_cluster_result(model_name, layer, n_clusters, size, X, random_seed, write_file=False):
    file_name = f'{model_name}_{n_clusters}_layer{layer}.npy'
    output_path = os.path.join(f'/usr/workdir/HeterExpert/Split/ilp_split/cluster_cache/{model_name}', file_name)
    if os.path.exists(output_path) and not write_file:
        clusters = np.load(output_path)
    else:
        # clusters = fcluster(linkage(X, method='ward'), n_clusters, criterion='maxclust') - 1  # idx range(1, NUM_MODULES+1) - 1
        # clusters = KMeans(n_clusters=n_clusters, random_state=random_seed).fit_predict(X)  # idx range(0, NUM_MODULES)
        clusters = KMeansConstrained(n_clusters=n_clusters, size_min=size, size_max=size, random_state=random_seed).fit_predict(X)  # idx range(0, NUM_MODULES)

        np.save(output_path, clusters)
        
    return clusters

def main():
    random_seed = 42
    set_seed(random_seed)

    model_name = 'llama3.2-1b'
    model_path = f'/usr/workdir/models/{model_name}'
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)

    data_path = f'/usr/workdir/HeterExpert/Neuron_Importance/score/cluster/{model_name}'
    domains_data = np.load(f'{data_path}/importance_score.npy')  # [num_layers, num_neurons, num_domains]
    
    num_layers, num_neurons, num_domains = domains_data.shape
    assert num_neurons % NUM_MODULES == 0
    size_module = num_neurons // NUM_MODULES
    domains_data_reduced = np.zeros((num_layers, NUM_MODULES, num_domains))
    module2neurons = np.zeros((num_layers, NUM_MODULES, size_module))

    for layer in range(num_layers):
        params = model.model.layers[layer].mlp.gate_proj.weight.data  # [dff_hidden_size, hidden_size]
        # params += model.model.layers[layer].mlp.up_proj.weight.data   # TODO
        params_norm = normalize(params, axis=1)
        
        clusters = get_cluster_result(model_name, layer, NUM_MODULES, size_module, params_norm, random_seed)
        
        clusters_idx_range = range(NUM_MODULES)
        
        experts_size = Counter([np.sum(clusters == i) for i in clusters_idx_range])
        assert sum(experts_size.values()) == NUM_MODULES
        experts_size_sorted = sorted(experts_size.items(), key=lambda x: x[0])
        print(f'Layer {layer}: {experts_size_sorted}')

        
        for cluster_idx in clusters_idx_range:
            neurons_idx = np.where(clusters == cluster_idx)[0]
            module2neurons[layer][cluster_idx] = neurons_idx
            cluster_scores = np.sum(domains_data[layer][neurons_idx], axis=0)
            domains_data_reduced[layer][cluster_idx] = cluster_scores
        
    np.savez(f'{data_path}/importance_score_reduced_{NUM_MODULES}.npz', domains_data_reduced=domains_data_reduced, module2neurons=module2neurons)

if __name__ == '__main__':
    main()