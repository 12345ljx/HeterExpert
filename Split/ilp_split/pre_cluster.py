from collections import defaultdict, Counter
import numpy as np
import torch
from transformers import set_seed, AutoModelForCausalLM
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.preprocessing import normalize
from k_means_constrained import KMeansConstrained

NUM_MODULES = 512

def main():
    random_seed = 42
    set_seed(random_seed)

    model_path = '/usr/workdir/models/llama3.2-1b'
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)

    data_path = '/usr/workdir/HeterExpert/Neuron_Importance/score5000'
    domains_data = np.load(f'{data_path}/importance_score(non_log).npy')  # [num_layers, num_neurons, num_domains]
    
    num_layers, num_neurons, num_domains = domains_data.shape
    assert num_neurons % NUM_MODULES == 0
    size_module = num_neurons // NUM_MODULES
    domains_data_reduced = np.zeros((num_layers, NUM_MODULES, num_domains))
    module2neurons = np.zeros((num_layers, NUM_MODULES, size_module))

    num_layers = model.config.num_hidden_layers

    for layer in range(num_layers):
        params = model.model.layers[layer].mlp.gate_proj.weight.data  # [dff_hidden_size, hidden_size]
        # params += model.model.layers[layer].mlp.up_proj.weight.data   # TODO
        params_norm = normalize(params, axis=1)
        
        # clusters = fcluster(linkage(params_norm, method='ward'), NUM_MODULES, criterion='maxclust')  # idx range(1, NUM_MODULES+1)
        # clusters = KMeans(n_clusters=NUM_MODULES, random_state=random_seed).fit_predict(params_norm)  # idx range(0, NUM_MODULES)
        clusters = KMeansConstrained(n_clusters=NUM_MODULES, size_min=size_module, size_max=size_module, random_state=random_seed).fit_predict(params_norm)  # idx range(0, NUM_MODULES)
        
        start_idx = 0  # 1 for fcluster
        clusters_idx_range = range(start_idx, NUM_MODULES + start_idx)
        
        experts_size = Counter([np.sum(clusters == i) for i in clusters_idx_range])
        assert sum(experts_size.values()) == NUM_MODULES
        experts_size_sorted = sorted(experts_size.items(), key=lambda x: x[0])
        print(f'Layer {layer}: {experts_size_sorted}')

        
        for cluster_idx in clusters_idx_range:
            neurons_idx = np.where(clusters == cluster_idx)[0]
            module2neurons[layer][cluster_idx] = neurons_idx
            cluster_scores = np.sum(domains_data[layer][neurons_idx], axis=0)
            domains_data_reduced[layer][cluster_idx - start_idx] = cluster_scores
        
    # np.save(f'{data_path}/importance_score_reduced.npy', domains_data_reduced)
    np.savez(f'{data_path}/importance_score_reduced_{NUM_MODULES}(non_log).npz', domains_data_reduced=domains_data_reduced, module2neurons=module2neurons)

if __name__ == '__main__':
    main()