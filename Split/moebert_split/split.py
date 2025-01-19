import os
import numpy as np
import random
import sys
sys.path.append('/usr/workdir/HeterExpert')
from Neuron_Importance.analyse_score import read_score, preprocess_score

def main():
    random_seed = 42
    random.seed(random_seed)
    
    model_name = 'llama3.2-1b'
    function_name = 'domains'
    DOMAIN_NUM = 8
    NUM_EXPERT = 8
    dff_hidden_size = 8192
    num_hidden_layers = 16
    size_expert = dff_hidden_size // NUM_EXPERT
    num_sharing = size_expert // 2

    data_path = '/usr/workdir/HeterExpert/Neuron_Importance/score'
    domains_data = read_score(num_hidden_layers, dff_hidden_size, DOMAIN_NUM, data_path)  # [num_layers, num_neurons, num_domains]
    domains_data = np.sum(domains_data, axis=2)  # [num_layers, num_neurons]
    
    for layer_idx in range(num_hidden_layers):
        neuron_score_pair = [(neuron_idx, score) for neuron_idx, score in enumerate(domains_data[layer_idx])]
        neuron_score_pair_sorted = sorted(neuron_score_pair, key=lambda x: x[1], reverse=True)
        neurons_sharing = [value[0] for value in neuron_score_pair_sorted[:num_sharing]]
        neurons_remaining = [neuron for neuron in range(dff_hidden_size) if neuron not in neurons_sharing]
    
        experts = [neurons_sharing[:] for _ in range(NUM_EXPERT)]
        for i in range(NUM_EXPERT):
            remain = neurons_remaining[i::NUM_EXPERT][:]
            random.shuffle(remain)
            experts[i].extend(remain)
            experts[i] = experts[i][:size_expert]
         
        neuron2experts = [[] for _ in range(dff_hidden_size)]   
        for i in range(NUM_EXPERT):
            for neuron in experts[i]:
                neuron2experts[neuron].append(str(i))
        
        output_path = f'/usr/workdir/HeterExpert/Split/model_split/moebert/{model_name}/{function_name}/{NUM_EXPERT}'
        os.makedirs(output_path, exist_ok=True)
        with open(f'{output_path}/{layer_idx}.part.{NUM_EXPERT}', 'w') as f:
            for neuron in range(dff_hidden_size):
                f.write('{}\n'.format(','.join(neuron2experts[neuron])))
                
        
if __name__ == '__main__':
    main()