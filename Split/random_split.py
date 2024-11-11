import numpy as np
from transformers import AutoConfig
import time
from collections import Counter, defaultdict
import random, os


model_name = "llama3.2-1b"
expert_num = 16


# random_seed = int(time.time())
random_seed = 42

model_path = f"/usr/workdir/MoEfication/models/{model_name}"
if model_name in ("llama-7b", "relu-llama2-7b") :
    config = AutoConfig.from_pretrained(model_path)
    dff_hidden_size = config.intermediate_size
    layers_number = config.num_hidden_layers
elif model_name == "t5-base" :
    config = AutoConfig.from_pretrained(model_path)
    dff_hidden_size = config.d_ff
    layers_number = config.num_hidden_layers
elif model_name == "llama3.2-1b" :
    dff_hidden_size = 8192
    layers_number = 16
else:
    raise NotImplementedError

np.random.seed(random_seed)
random.seed(random_seed)

# generate the neurons in each expert
experts_split = dict()
if model_name == 't5-base':
    for layer in range(layers_number):
        experts_split[f'encoder_{layer}'] = np.random.randint(low=0, high=expert_num, size=dff_hidden_size)
    for layer in range(layers_number):
        experts_split[f'decoder_{layer}'] = np.random.randint(low=0, high=expert_num, size=dff_hidden_size)
elif model_name in ["relu-llama2-7b", "llama3.2-1b"]:
    for layer in range(layers_number):
        experts_split[f'{layer}'] = np.random.randint(low=0, high=expert_num, size=dff_hidden_size)
else:
    raise NotImplementedError

# average the number of neurons in each experts
for layer, module in experts_split.items():
    d = defaultdict(list)
    for node_idx, expert_idx in enumerate(module):
        d[expert_idx].append(node_idx)
        
    assert dff_hidden_size % expert_num == 0
    average_num = dff_hidden_size // expert_num
    
    need_move = list()
    for expert_idx, expert_nodes in d.items():
        if len(expert_nodes) > average_num:
            random.shuffle(expert_nodes)
            for i in range(average_num, len(expert_nodes)):
                need_move.append(expert_nodes[i])
            d[expert_idx] = expert_nodes[:average_num]

    random.shuffle(need_move)
    for expert_idx, expert_nodes in d.items():
        if len(expert_nodes) < average_num:
            pos = average_num-len(expert_nodes)
            expert_nodes += need_move[:pos]
            need_move = need_move[pos:]
        for node in expert_nodes:
            module[node] = expert_idx

# check the number of neurons in each experts 
for layer, module in experts_split.items(): 
    d = defaultdict(set)
    for node_idx, expert_idx in enumerate(module):
        d[expert_idx].add(node_idx)
    for expert_node in d.values():
        assert len(expert_node) == average_num

output_path = f"/usr/workdir/HeterExpert/Split/model_split/random/{model_name}/{expert_num}"
os.makedirs(output_path, exist_ok=True)
# write to file
for layer, module in experts_split.items():
    file_name = f'{layer}.part.{expert_num}'
    file_path = os.path.join(output_path, file_name)
    with open(file_path, 'w') as fp:
        for i in module:
            fp.write(str(i) + '\n')
    print(layer, Counter(module))
    
