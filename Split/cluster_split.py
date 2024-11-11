import os
from collections import Counter
import numpy as np
import torch
from k_means_constrained import KMeansConstrained
from transformers import set_seed, AutoModelForCausalLM
from sklearn.preprocessing import normalize

NUM_EXPERT = 16

def main():
    random_seed = 42
    set_seed(random_seed)

    model_name = 'llama3.2-1b'
    model_path = f'/usr/workdir/models/{model_name}'
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
    num_layers = model.config.num_hidden_layers
    num_neurons = model.config.intermediate_size
    assert num_neurons % NUM_EXPERT == 0
    size_module = num_neurons // NUM_EXPERT
    
    for layer_idx in range(num_layers):
        params = model.model.layers[layer_idx].mlp.gate_proj.weight.data  # [dff_hidden_size, hidden_size]
        # params += model.model.layers[layer_idx].mlp.up_proj.weight.data   # TODO
        params_norm = normalize(params, axis=1)
        
        # clusters = fcluster(linkage(params_norm, method='ward'), NUM_EXPERT, criterion='maxclust')  # idx range(1, NUM_EXPERT+1)
        # clusters = KMeans(n_clusters=NUM_EXPERT, random_state=random_seed).fit_predict(params_norm)  # idx range(0, NUM_EXPERT)
        clusters = KMeansConstrained(n_clusters=NUM_EXPERT, size_min=size_module, size_max=size_module, random_state=random_seed).fit_predict(params_norm)  # idx range(0, NUM_EXPERT)
        
        start_idx = 0  # 1 for fcluster
        clusters_idx_range = range(start_idx, NUM_EXPERT + start_idx)
        
        experts_size = Counter([np.sum(clusters == i) for i in clusters_idx_range])
        assert sum(experts_size.values()) == NUM_EXPERT

        neuron_expert_pairs = []
        for neuron_idx in range(num_neurons):
            neuron_expert_pairs.append((neuron_idx, clusters[neuron_idx] - start_idx))
        
        output_path = f'/usr/workdir/HeterExpert/Split/model_split/cluster/{model_name}/{NUM_EXPERT}'
        os.makedirs(output_path, exist_ok=True)
        with open(f'{output_path}/{layer_idx}.part.{NUM_EXPERT}', 'w') as f:
            for neuron, expert in neuron_expert_pairs:
                f.write(f'{expert}\n')
        
if __name__ == '__main__':
    main()