import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import re

from Eval.eval_main import eval_lora, get_model_tokenizer, Train2Eval
from MoEfication.moefication import moeficate, MoEArgs
from tools.path_lib import get_gate_path_smart

def main():
    model_name = 'llama3.2-1b'
    model_path = f'./models/{model_name}'
    model, tokenizer = get_model_tokenizer(model_path)
    
    apply_chat_template = False
    num_fewshot = 0
    
    # ['arc_easy', 'arc_challenge', 'piqa', 'openbookqa', 'winogrande', 'sciq', 'siqa', 'alpaca_cleaned']
    task_name = 'arc_easy'
    eval_task_name = Train2Eval.get(task_name, task_name)
    write_out = False
    
    moeargs = MoEArgs(
        model_name=model_name,
        function_name='domains(r4l2)',
        split_mode='ilp',  # cluster, ilp
        gate_mode='top_k',  # top_k, top_p, dynk_max
        gate_backend=None,  # None, index, triton
        static=False,
        begin_layer=4,
        end_layer=15,
        num_expert=8,
        num_selected_expert=4,
        # top_p_threshold=0.66,
        # tau=0.18,
        balance_loss_weight=5e-4,
    )
    
    model = moeficate(model, moeargs)
    
    
    # gate_path = get_gate_path(task_name=task_name, **moeargs.to_dict())
    gate_path = get_gate_path_smart(extra_elements=['seed=42'], task_name=task_name, **moeargs.to_dict())
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
    
    print('the gate path is:', gate_path)
    

if __name__ == '__main__':
    main()