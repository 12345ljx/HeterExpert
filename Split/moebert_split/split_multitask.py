import os
import random
from Neuron_Score.process_score import read_score

def main():
    random_seed = 42
    random.seed(random_seed)
    
    model_name = 'llama3.2-1b'
    function_name = 'domains'
    DOMAIN_NUM = 8
    NUM_EXPERT = 8
    assert NUM_EXPERT == DOMAIN_NUM, "NUM_EXPERT should be equal to DOMAIN_NUM, because we assign one domain to one expert"
    dff_hidden_size = 8192
    num_hidden_layers = 16
    size_expert = dff_hidden_size // NUM_EXPERT

    data_path = f'./Neuron_Score/score/cluster/{model_name}'
    domains_data = read_score(num_hidden_layers, dff_hidden_size, DOMAIN_NUM, data_path)  # [num_layers, num_neurons, num_domains]
    
    for layer_idx in range(num_hidden_layers):
        neuron_score_pair = [(neuron_idx, score) for neuron_idx, score in enumerate(domains_data[layer_idx])]
        experts = [[] for _ in range(NUM_EXPERT)]
        for i in range(NUM_EXPERT):
            neuron_score_pair_sorted = sorted(neuron_score_pair, key=lambda x: x[1][i], reverse=True)
            experts[i] = [value[0] for value in neuron_score_pair_sorted[:size_expert]]
            
        neuron2experts = [[] for _ in range(dff_hidden_size)]   
        for i in range(NUM_EXPERT):
            for neuron in experts[i]:
                neuron2experts[neuron].append(str(i))
        
        output_path = f'./Split/model_split/moebert_multitask/{model_name}/{function_name}/{NUM_EXPERT}'
        os.makedirs(output_path, exist_ok=True)
        with open(f'{output_path}/{layer_idx}.part.{NUM_EXPERT}', 'w') as f:
            for neuron in range(dff_hidden_size):
                f.write('{}\n'.format(','.join(neuron2experts[neuron])))
                
        
if __name__ == '__main__':
    main()