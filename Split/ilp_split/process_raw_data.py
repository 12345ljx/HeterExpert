import os
import numpy as np

NUM_MODULES = 128
NUM_HIDDEN_LAYERS = 16
NUM_EXPERT = 16
NUM_EXPERT_ACT = 8

model_name = 'llama3.2-1b'
function_name = 'domains'

base_path = f'/usr/workdir/HeterExpert/Split/model_split/ilp/{model_name}/{function_name}(m=128)/{NUM_EXPERT}'
os.makedirs(base_path, exist_ok=True)

data = np.load(f'/usr/workdir/HeterExpert/Neuron_Importance/score5000/importance_score_reduced_{NUM_MODULES}.npz')
module2neurons = data['module2neurons'].astype(np.int32)  # [num_layers, num_modules, num_neurons]

for layer_idx in range(NUM_HIDDEN_LAYERS):
    placement = np.load(f'/usr/workdir/HeterExpert/Split/ilp_split/raw_data/llama3.2-1b/domains/k{NUM_EXPERT_ACT}n{NUM_EXPERT}m{NUM_MODULES}/neuron_grouping.layer{layer_idx}.npz')['placement']  # [num_neurons,]

    neuron_expert_pairs = []
    for expert_idx in range(NUM_EXPERT):
        modules = np.where(placement == expert_idx)[0]
        for module in modules:
            neurons = module2neurons[layer_idx][module]
            for neuron in neurons:
                neuron_expert_pairs.append((neuron, expert_idx))
    
    neuron_expert_pairs = sorted(neuron_expert_pairs, key=lambda x: x[0])
    
    with open(f'{base_path}/{layer_idx}.part.{NUM_EXPERT}', 'w') as f:
        for neuron, expert in neuron_expert_pairs:
            f.write(f'{expert}\n')
