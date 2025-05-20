import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from enum import Enum
import re

import torch
from eval_main import analyze_param, eval_lora_ppls, eval_on_tasks, eval_lora, get_model_tokenizer, get_model_path, Train2Eval
import sys
sys.path.append('/usr/workdir/MoEfication/moefication')
from moefication import moeficate, prune, load_gates, MoEArgs
from tools.path_lib import get_gate_path, get_lora_path, get_gate_path_smart

FUNCTION_DICT = {
    'relu-chatglm3-6b' : 'harness',
    'relu-llama2-7b': 'harness(down_proj_abs)',
    'llama3.2-1b': 'domains(r4l2)',  # XXX
    'llama3.2-1b-instruct': 'domains(module_stable)',
    'llama3.2-3b': 'domains(r4l2)',
    'relu-llama2-7b': 'domains(module_stable)',
    'SparseQwen2-7B': 'domains(module_stable)',
}

class TrainMode(Enum):
    CO_TRAIN=1
    STEP_TRAIN=2

def main():
    # torch.cuda.set_device(1)
    model_name = 'llama3.2-1b'  # ['relu-llama2-7b', 'llama-7b', 'llama2-7b', 't5-base', 'relu-chatglm3-6b', 'llama3.2-1b', 'SparseQwen2-7B']
    model_path = get_model_path(model_name)
    model, tokenizer = get_model_tokenizer(model_path)
    model_param_ori, ffn_param_ori = analyze_param(model)
    
    # apply_chat_template = True if tokenizer.chat_template else False
    apply_chat_template = False
    num_fewshot = 0
    
    # ['arc_easy', 'arc_challenge', 'piqa', 'openbookqa', 'winogrande', 'sciq', 'siqa', 'alpaca_cleaned']
    task_name = 'arc_challenge'
    eval_task_name = Train2Eval.get(task_name, task_name)
    write_out = False
    
    training_mode = TrainMode.CO_TRAIN
    
    moeargs = MoEArgs(
        model_name=model_name,
        function_name='domains(r4l2)',
        split_mode='cluster',  # cluster, ilp
        gate_mode='top_k',  # top_k, top_p, dynk_max
        gate_backend=None,  # None, index, triton
        static=False,
        begin_layer=4,
        end_layer=15,
        num_expert=8,
        num_selected_expert=1,
        # top_p_threshold=0.66,
        # tau=0.18,
        # gamma=0.1,
        balance_loss_weight=5e-4,
    )
    
    model = moeficate(model, moeargs)
    # model.to(torch.bfloat16)
    
    # for param_name, param in model.named_parameters():
    #     print(f"{param_name}, {param.size()}, {param.dtype}, requires_grad: {param.requires_grad}, {param.device}")
    
    if training_mode == TrainMode.STEP_TRAIN:
        load_gates(model, model_name, task_name, moeargs)
        # model = prune(model)
        result_prune = eval_on_tasks(
            model=model, 
            tokenizer=tokenizer, 
            task_name=eval_task_name, 
            write_out=write_out, 
            model_name=model_name, 
            num_fewshot=num_fewshot,
            apply_chat_template=apply_chat_template,
        )
        
        lora_path = get_lora_path(get_gate_path(task_name=task_name, **moeargs.to_dict()))
        if lora_path:
            base_lora_path, last_checkpoint = os.path.split(lora_path)
            end_step = int(re.search(r"\d+", last_checkpoint)[0])
            lora_range = range(end_step, end_step + 1, 20)
            results_list = eval_lora(model, tokenizer, eval_task_name, base_lora_path, lora_range, write_out, model_name)
            
        if result_prune:
            print(f'original pruned model: {result_prune}')
    elif training_mode == TrainMode.CO_TRAIN:
        # gate_path = get_gate_path(task_name=task_name, **moeargs.to_dict())
        gate_path = get_gate_path_smart(extra_elements=['seed=43'], task_name=task_name, **moeargs.to_dict())
        base_lora_path, last_checkpoint = os.path.split(gate_path)
        end_step = int(re.search(r"\d+", last_checkpoint)[0])
        lora_range = list(range(200, end_step, 200))
        if end_step not in lora_range:
            lora_range.append(end_step)
            
        results_list = eval_lora(
            model=model, 
            tokenizer=tokenizer, 
            task_name=eval_task_name,
            base_lora_path=base_lora_path,
            lora_range=lora_range, 
            write_out=write_out, 
            model_name=model_name,
            num_fewshot=num_fewshot,
            apply_chat_template=apply_chat_template
            )
        
        # eval_lora_ppls(
        #     model=model, 
        #     tokenizer=tokenizer, 
        #     base_lora_path=base_lora_path,
        #     lora_range=lora_range
        # )
        
        print('the gate path is:', gate_path)
        

    model_param_new, ffn_param_new = analyze_param(model)
    print('whole model sparsity: {:.5f}'.format(model_param_new / model_param_ori))
    print('ffn sparsity: {:.5f}'.format(ffn_param_new / ffn_param_ori))
    

if __name__ == '__main__':
    main()