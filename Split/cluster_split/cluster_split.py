import os
import sys
from collections import Counter
import numpy as np
import torch
from k_means_constrained import KMeansConstrained
from transformers import set_seed, AutoModelForCausalLM
from sklearn.preprocessing import normalize

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

def main(num_expert):
    random_seed = 42
    set_seed(random_seed)

    model_name = 'SparseQwen2-7B'
    model_path = f'/usr/workdir/models/{model_name}'
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, trust_remote_code=True)
    num_layers = model.config.num_hidden_layers
    num_neurons = model.config.intermediate_size
    assert num_neurons % num_expert == 0
    size_expert = num_neurons // num_expert
    
    for layer_idx in range(num_layers):
        params = model.model.layers[layer_idx].mlp.gate_proj.weight.data  # [dff_hidden_size, hidden_size]
        # params += model.model.layers[layer_idx].mlp.up_proj.weight.data   # TODO
        params_norm = normalize(params, axis=1)
        
        clusters = get_cluster_result(model_name, layer_idx, num_expert, size_expert, params_norm, random_seed)
        
        experts_size = Counter([np.sum(clusters == i) for i in range(num_expert)])
        assert sum(experts_size.values()) == num_expert

        neuron_expert_pairs = [(neuron_idx, clusters[neuron_idx]) for neuron_idx in range(num_neurons)]
        
        output_path = f'/usr/workdir/HeterExpert/Split/model_split/cluster/{model_name}/{num_expert}'
        os.makedirs(output_path, exist_ok=True)
        with open(f'{output_path}/{layer_idx}.part.{num_expert}', 'w') as f:
            for neuron, expert in neuron_expert_pairs:
                f.write(f'{expert}\n')
        
if __name__ == '__main__':
    num_expert = 8
    if len(sys.argv) > 1:
        num_expert = int(sys.argv[1])
        
    print(f'split model to {num_expert} experts')
    main(num_expert)