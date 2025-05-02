import numpy as np
from transformers import AutoConfig
import time
from collections import Counter, defaultdict
import random, os

def get_model_info(model_name):
    model_path = f"/usr/workdir/MoEfication/models/{model_name}"
    if not os.path.exists(model_path):
        model_path = f"/usr/workdir/models/{model_name}"
        
    if 'llama' in model_name:
        config = AutoConfig.from_pretrained(model_path)
        dff_hidden_size = config.intermediate_size
        layers_number = config.num_hidden_layers
    elif model_name == "t5-base" :
        config = AutoConfig.from_pretrained(model_path)
        dff_hidden_size = config.d_ff
        layers_number = config.num_hidden_layers
    else:
        raise NotImplementedError

    return dff_hidden_size, layers_number

def get_random_split(model_name, layers_number, expert_num, dff_hidden_size):
    # generate the neurons in each expert
    experts_split = dict()
    if model_name == 't5-base':
        for layer in range(layers_number):
            experts_split[f'encoder_{layer}'] = np.random.randint(low=0, high=expert_num, size=dff_hidden_size)
        for layer in range(layers_number):
            experts_split[f'decoder_{layer}'] = np.random.randint(low=0, high=expert_num, size=dff_hidden_size)
    elif 'llama' in model_name:
        for layer in range(layers_number):
            experts_split[f'{layer}'] = np.random.randint(low=0, high=expert_num, size=dff_hidden_size)
    else:
        raise NotImplementedError
    
    return experts_split

def averate_correction(experts_split, average_num):
    # average the number of neurons in each experts
    for layer, placement in experts_split.items():
        tmp_dict = defaultdict(list)
        for node_idx, expert_idx in enumerate(placement):
            tmp_dict[expert_idx].append(node_idx)
            
        need_move = list()
        for expert_idx, expert_nodes in tmp_dict.items():
            if len(expert_nodes) > average_num:
                random.shuffle(expert_nodes)
                need_move.extend(expert_nodes[average_num:len(expert_nodes)])
                tmp_dict[expert_idx] = expert_nodes[:average_num]

        random.shuffle(need_move)
        for expert_idx, expert_nodes in tmp_dict.items():
            if len(expert_nodes) < average_num:
                pos = average_num - len(expert_nodes)
                expert_nodes += need_move[:pos]
                need_move = need_move[pos:]
                
        for expert_idx, expert_nodes in tmp_dict.items():
            for node in expert_nodes:
                placement[node] = expert_idx
                
def check_range(experts_split, min_num, max_num):
    # check the number of neurons in each experts 
    for layer, placement in experts_split.items(): 
        tmp_dict = defaultdict(set)
        for node_idx, expert_idx in enumerate(placement):
            tmp_dict[expert_idx].add(node_idx)
        for expert_node in tmp_dict.values():
            assert len(expert_node) <= max_num and len(expert_node) >= min_num
            
def write_result(output_path, experts_split, expert_num):
    # write to file
    os.makedirs(output_path, exist_ok=True)
    for layer, placement in experts_split.items():
        file_name = f'{layer}.part.{expert_num}'
        file_path = os.path.join(output_path, file_name)
        with open(file_path, 'w') as fp:
            for i in placement:
                fp.write(str(i) + '\n')
        print(layer, Counter(placement))
        
def main():
    random_seed = 42
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    model_name = "llama3.2-1b-instruct"
    expert_num = 8
    
    dff_hidden_size, layers_number = get_model_info(model_name)
    experts_split = get_random_split(model_name, layers_number, expert_num, dff_hidden_size)
    
    assert dff_hidden_size % expert_num == 0
    average_num = dff_hidden_size // expert_num
    averate_correction(experts_split, average_num)
    check_range(experts_split, average_num, average_num) 
    
    output_path = f"/usr/workdir/MoEfication/moefication/get_hidden/model_split/random/{model_name}/{expert_num}"
    write_result(output_path, experts_split, expert_num)
    
if __name__ == "__main__":
    main()
