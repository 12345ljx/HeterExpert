import os
from typing import Optional

def path_filter(root_dir: str, target_strs: list):
    if not os.path.exists(root_dir):
        raise FileExistsError(f"Path {root_dir} does not exist!")
    
    path_list = []
    for root, dirs, files in os.walk(root_dir):
        if len(dirs) == 0 or not dirs[0].isdigit():
            continue
        path_list.append(root)
    
    for target_str in target_strs:
        path_list = [x for x in path_list if target_str in x]
    return path_list

def find_time_id(start_id, root_dir: str) -> str:
    ids = [int(file) for file in os.listdir(root_dir) if file.isdigit()]
    ids = [id for id in ids if id >= start_id]
    assert len(ids) == 1, f"Find ids {ids} in {root_dir}"
    return os.path.join(root_dir, str(ids[0]))

def find_last_checkpoint(root_dir: str) -> str:
    checkpoints = [int(file.split("-")[-1]) for file in os.listdir(root_dir) if file.startswith("checkpoint-")]
    assert len(checkpoints) > 0, f"Can't find checkpoint in {root_dir}"
    gate_path = os.path.join(root_dir, f"checkpoint-{max(checkpoints)}")
    return gate_path

def get_gate_path_smart(extra_elements: Optional[list] = None, **kwargs):
    base_path = "/usr/workdir/HeterExpert/models/parts/gate,lora"
    
    if kwargs['split_mode'] in ['cluster', 'random', 'random_hetero']:
        split_mode = kwargs['split_mode']
    else:
        split_mode = f"{kwargs['split_mode']}[{kwargs['function_name']}]"
        
    elements = [
        f"{kwargs['model_name']}:{kwargs['task_name']}",
        split_mode + ',',
        kwargs['gate_mode'],
        f"{kwargs['begin_layer']}-{kwargs['end_layer']}",
        f"balance_loss_weight={kwargs['balance_loss_weight']}",
    ]
    
    if kwargs['gate_mode'] == 'top_k':
        elements.append(f"n{kwargs['num_expert']}k{kwargs['num_selected_expert']}" + "(")
    elif kwargs['gate_mode'] == 'top_p':
        elements.append(f"n{kwargs['num_expert']}p{kwargs['top_p_threshold']}" + "(")
    elif kwargs['gate_mode'] == 'dynk_max':
        elements.append(f"n{kwargs['num_expert']}t{kwargs['tau']}" + "(")
    else:
        raise ValueError(f"Unknown gate mode {kwargs['gate_mode']}")
    
    if extra_elements and len(extra_elements) > 0:
        elements += extra_elements
    path_candidate = path_filter(base_path, elements)
    
    if len(path_candidate) == 0:
        raise FileExistsError("Can't find the weight file!")
    elif len(path_candidate) > 1:
        raise ValueError("Multiple gate paths found!")
    
    base_gate_path = path_candidate[0]
    gate_path = find_time_id(1746173318, base_gate_path)
    gate_path = find_last_checkpoint(gate_path)
    print("loading gate from: ", gate_path)
    return gate_path
    