import os
import random
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

import sys
sys.path.append('/usr/workdir/MoEfication/moefication')
from moefication import moeficate, MoEArgs
from get_labels import get_labels_base_path, read_labels
from gates_analysis.gate_analyse_dynamic import load_data

from ExpertObserve import ExpertObserve

def get_moe_model(moeargs):
    model_path = f'/usr/workdir/models/{moeargs.model_name}'
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.convert_ids_to_tokens(0)
    model = moeficate(model, moeargs)
    return model, tokenizer

def get_relative_expert_size(config, moeargs):
    num_expert = moeargs.num_expert
    experts_size = np.zeros((config.num_hidden_layers, num_expert)) 
    base_path = moeargs.get_labels_base_path()
    for layer in moeargs.layer_range:
        labels = MoEArgs.read_labels(os.path.join(base_path, f'{num_expert}/{layer}.part.{num_expert}'))
        for i in range(num_expert):
            experts_size[layer][i] = len(labels[i]) / config.intermediate_size
        assert np.sum(experts_size[layer]) == 1
    return experts_size

def save_data(data):
    import csv
    with open('/usr/workdir/HeterExpert/tools/activated_params/activated_params_3.csv', 'w') as f:
        writer = csv.writer(f)
        for k, v in data.items():
            writer.writerow([k, v])

def main():
    model_name = 'llama3.2-1b'
    function_name = 'domains(module_stable)'
    task_name = 'arc_easy'
    moeargs = MoEArgs(
        model_name=model_name,
        function_name=function_name,
        split_mode='ilp',
        gate_mode='top_k',
        static=False,
        begin_layer=4,
        end_layer=15,
        num_expert=8,
        num_selected_expert=4,
        # top_p_threshold=0.64,
        # tau=0.9,
        balance_loss_weight=5e-4,
    )
    
    model, tokenizer = get_moe_model(moeargs)
    relative_expert_size = get_relative_expert_size(model.config, moeargs)
    observer = ExpertObserve(relative_expert_size)
    observe_function = observer.observe_library(model_name)
    
    sentences = load_data(task_name)
    sentences = random.sample(sentences, min(5000, len(sentences)))
    sentences = sorted(sentences, key=len, reverse=True)
    model.eval().cuda()
    batch_size = 4
    
    token_params = defaultdict(list)
    for index in tqdm(range(len(sentences) // batch_size + 1)) :
        batch_sentences = sentences[index * batch_size : (index + 1) * batch_size]
        if len(batch_sentences) == 0 :
            continue
        
        encoded_input = tokenizer(batch_sentences, return_tensors="pt", padding=True, truncation=True, max_length=1024)
        input_ids = encoded_input.input_ids.cuda()
        attention_mask = encoded_input.attention_mask.cuda()
        batch_size_true, sequence_length = input_ids.shape
        num_layer = model.config.num_hidden_layers
        res = np.zeros((num_layer, batch_size_true, sequence_length))
        res = res.reshape(num_layer, -1)
        with torch.no_grad() :
            hooks = observe_function(model, res, attention_mask)
            model(input_ids=input_ids, attention_mask=attention_mask)
            observer.erase_hooks(hooks)
        res = res.reshape(num_layer, batch_size_true, sequence_length)
        
        for sentence_id in range(batch_size_true):
            tokens = tokenizer.convert_ids_to_tokens(input_ids[sentence_id])
            for token_id, token in enumerate(tokens):
                if token == tokenizer.pad_token:
                    continue
                for layer in moeargs.layer_range:
                    if layer != 3: continue
                    token_params[token].append(res[layer][sentence_id][token_id])
            
    token_params_avg = {k: sum(v)/len(v) for k, v in token_params.items()}
    token_params_avg = dict(sorted(token_params_avg.items(), key=lambda x: x[1]))
    
    print(token_params_avg)
    
    save_data(token_params_avg)
    
    

if __name__ == '__main__':
    main()