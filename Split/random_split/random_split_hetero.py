import numpy as np
import random
from collections import defaultdict

from random_split_homo import get_model_info, get_random_split, check_range, write_result
import sys
sys.path.append('/usr/workdir/MoEfication/moefication')
from get_labels import read_labels

def get_ilp_expert_size(model_name, layers_number, expert_num):
    ilp_base_path = f'/usr/workdir/HeterExpert/Split/model_split/ilp/{model_name}/domains(module_stable)/{expert_num}'
    experts_size = {}
    for layer_idx in range(layers_number):
        labels = read_labels(f'{ilp_base_path}/{layer_idx}.part.{expert_num}')
        experts_size[layer_idx] = [len(label) for label in labels]
    return experts_size

def expert_size_correction(experts_split, experts_size):
    for layer, placement in experts_split.items():
        expert_size = experts_size[int(layer)]
        tmp_dict = defaultdict(list)
        for node_idx, expert_idx in enumerate(placement):
            tmp_dict[expert_idx].append(node_idx)
            
        need_move = list()
        for expert_idx, expert_nodes in tmp_dict.items():
            correct_size = expert_size[expert_idx]
            if len(expert_nodes) > correct_size:
                random.shuffle(expert_nodes)
                need_move.extend(expert_nodes[correct_size:len(expert_nodes)])
                tmp_dict[expert_idx] = expert_nodes[:correct_size]

        random.shuffle(need_move)
        for expert_idx, expert_nodes in tmp_dict.items():
            correct_size = expert_size[expert_idx]
            if len(expert_nodes) < correct_size:
                pos = correct_size - len(expert_nodes)
                expert_nodes += need_move[:pos]
                need_move = need_move[pos:]
                
        for expert_idx, expert_nodes in tmp_dict.items():
            for node in expert_nodes:
                placement[node] = expert_idx

def main():
    random_seed = 42
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    model_name = "llama3.2-1b-instruct"
    expert_num = 8
    
    dff_hidden_size, layers_number = get_model_info(model_name)
    experts_split = get_random_split(model_name, layers_number, expert_num, dff_hidden_size)
    
    ilp_experts_size = get_ilp_expert_size(model_name, layers_number, expert_num)
    expert_size_correction(experts_split, ilp_experts_size)
    
    assert dff_hidden_size % expert_num == 0
    average_num = dff_hidden_size // expert_num
    check_range(experts_split, average_num * 0.5, average_num * 2) 
    
    output_path = f"/usr/workdir/HeterExpert/Split/model_split/random_hetero/{model_name}/{expert_num}"
    write_result(output_path, experts_split, expert_num)
    
if __name__ == '__main__':
    main()