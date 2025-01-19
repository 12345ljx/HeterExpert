import os
import sys
from collections import Counter
import numpy as np
import torch
from k_means_constrained import KMeansConstrained
from transformers import set_seed, AutoModelForCausalLM
from sklearn.preprocessing import normalize

def main(num_expert):
    random_seed = 42
    set_seed(random_seed)

    model_name = 'llama3.2-1b'
    model_path = f'/usr/workdir/models/{model_name}'
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
    num_layers = model.config.num_hidden_layers
    num_neurons = model.config.intermediate_size
    assert num_neurons % num_expert == 0
    size_expert = num_neurons // num_expert
    
    for layer_idx in range(num_layers):
        params = model.model.layers[layer_idx].mlp.gate_proj.weight.data  # [dff_hidden_size, hidden_size]
        # params += model.model.layers[layer_idx].mlp.up_proj.weight.data   # TODO
        params_norm = normalize(params, axis=1)
        
        # clusters = fcluster(linkage(params_norm, method='ward'), num_expert, criterion='maxclust')  # idx range(1, num_expert+1)
        # clusters = KMeans(n_clusters=num_expert, random_state=random_seed).fit_predict(params_norm)  # idx range(0, num_expert)
        clusters = KMeansConstrained(n_clusters=num_expert, size_min=size_expert, size_max=size_expert, random_state=random_seed).fit_predict(params_norm)  # idx range(0, num_expert)
        
        start_idx = 0  # 1 for fcluster
        clusters_idx_range = range(start_idx, num_expert + start_idx)
        
        experts_size = Counter([np.sum(clusters == i) for i in clusters_idx_range])
        assert sum(experts_size.values()) == num_expert

        neuron_expert_pairs = []
        for neuron_idx in range(num_neurons):
            neuron_expert_pairs.append((neuron_idx, clusters[neuron_idx] - start_idx))
        
        output_path = f'/usr/workdir/HeterExpert/Split/model_split/cluster/{model_name}/{num_expert}'
        os.makedirs(output_path, exist_ok=True)
        with open(f'{output_path}/{layer_idx}.part.{num_expert}', 'w') as f:
            for neuron, expert in neuron_expert_pairs:
                f.write(f'{expert}\n')
        
if __name__ == '__main__':
    num_expert = 16
    if len(sys.argv) > 1:
        num_expert = int(sys.argv[1])
        
    print(f'split model to {num_expert} experts')
    main(num_expert)