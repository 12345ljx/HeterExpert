import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import sys
import re
import json
from collections.abc import Mapping
from collections import defaultdict
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, AutoModel
from peft import PeftModel
from tqdm import tqdm

from eval_ppl import eval_ppl
from eval_few_shot import eval_few_shot

sys.path.append('/usr/workdir/MoEfication/moefication')
from tools.analyse_params import count_ffn_param, count_parameters, ffn_ratio

Train2Eval = {
    'arc_mix': ['arc_easy', 'arc_challenge'],
    'siqa': 'social_iqa',
    'bigbench_abstract_narrative_understanding': 'bigbench_abstract_narrative_understanding_multiple_choice',
    'bigbench_goal_step_wikihow': 'bigbench_goal_step_wikihow_multiple_choice',
    'bigbench_social_iqa': 'bigbench_social_iqa_multiple_choice',
    'bigbench_timedial': 'bigbench_timedial_multiple_choice',
    'mmlu_professional_law': 'mmlu_international_law',
    'mmlu_high_school_world_history': 'mmlu_high_school_us_history',
    'mmlu_us_foreign_policy': 'mmlu_high_school_government_and_politics',
    'mmlu_high_school_computer_science': 'mmlu_college_computer_science',
    'alpaca_cleaned': ['arc_easy', 'arc_challenge', 'piqa', 'openbookqa', 'winogrande', 'sciq', 'social_iqa'],
}

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        try:
            return super(NumpyEncoder, self).default(obj)
        except TypeError:
            return repr(obj)
        
def get_metric_result(results: dict):
    if 'acc_norm,none' in results:
        return results['acc_norm,none']
    else:
        return results['acc,none']

def eval_on_tasks(model, tokenizer, task_name, write_out=False, model_name=None, num_fewshot=None, apply_chat_template=False, postfix=""):
    if not torch.cuda.is_available() :
        raise NotImplementedError
    model.cuda().eval()
    
    task_list = task_name if isinstance(task_name, list) else [task_name]
    results = eval_few_shot(model, tokenizer, task_list, num_fewshot=num_fewshot, apply_chat_template=apply_chat_template)  # fake data parallelism
    print("********************************")
    print(f"{num_fewshot}_shot evaluation results")
    print('results:', results['results'])
    
    if write_out:
        results_path = f"/usr/workdir/MoEfication/eval-harness/results/{model_name}:{task_name}{postfix}"
        os.makedirs(results_path, exist_ok=True)
        with open(os.path.join(results_path, "acc.json"), "w") as fp:
            json.dump(results, fp, indent=4, cls=NumpyEncoder)
            
    return {task: results['results'][task] for task in task_list}

def eval_lora(model, tokenizer, task_name, base_lora_path, lora_range, write_out=False, model_name=None, num_fewshot=None, apply_chat_template=False):
    results_list = []
    try:
        for step in tqdm(lora_range):
            print(f"lora step-{step}")
            model = PeftModel.from_pretrained(model, os.path.join(base_lora_path, f"checkpoint-{step}"))
            # model = model.merge_and_unload(progressbar=True, safe_merge=True, adapter_names=['default'])
            results_list.append(
                eval_on_tasks(
                    model=model,
                    tokenizer=tokenizer,
                    task_name=task_name, 
                    write_out=write_out, 
                    model_name=model_name,
                    num_fewshot=num_fewshot,
                    apply_chat_template=apply_chat_template,
                    postfix=f"_lora_{step}"
                    )
                )
            model = model.unload()
    except Exception as e:
        print(f"Error at step-{step}: {e}")
    finally:
        res_dict = defaultdict(list)
        for step, res in zip(lora_range, results_list):
            print("-"*8, f"step-{step}", "-"*8, sep="")
            for key, value in res.items():
                res_dict[key].append((f"step-{step}", get_metric_result(value)))
                print(f'{key}:\t\t{get_metric_result(value):.5f}')
        
        print("-"*8, f"best res", "-"*8, sep="")
        for key, value in res_dict.items():
            print(f'{key}:', end=' ')
            value_sorted = sorted(value, key=lambda x: x[1], reverse=True)
            max_value = value_sorted[0]
            print(f'{max_value[1]:.5f} ({max_value[0]})')
        
    return results_list

def eval_ppls(model, tokenizer):
    ppl_data_list = ['wikitext2', 'c4', 'alpaca', 'ptb']
    ppl_res = {data: eval_ppl(model, tokenizer, data) for data in ppl_data_list}
    # for data, res in ppl_res.items():
    #     print(f"{data} perplexity {res:.5f}")
    
    return ppl_res

def eval_lora_ppls(model, tokenizer, base_lora_path, lora_range):
    results_list = []
    try:
        for step in tqdm(lora_range):
            print(f"lora step-{step}")
            model = PeftModel.from_pretrained(model, os.path.join(base_lora_path, f"checkpoint-{step}"))
            # model = model.merge_and_unload(progressbar=True, safe_merge=True, adapter_names=['default'])
            results_list.append(eval_ppls(model, tokenizer))
            model = model.unload()
    except Exception as e:
        print(f"Error at step-{step}: {e}")
    finally:
        res_dict = defaultdict(list)
        for step, res in zip(lora_range, results_list):
            print("-"*8, f"step-{step}", "-"*8, sep="")
            for key, value in res.items():
                res_dict[key].append((f"step-{step}", value))
                print(f'{key}:\t\t{value:.5f}')
        
        print("-"*8, f"best res", "-"*8, sep="")
        for key, value in res_dict.items():
            print(f'{key}:', end=' ')
            value_sorted = sorted(value, key=lambda x: x[1], reverse=False)
            max_value = value_sorted[0]
            print(f'{max_value[1]:.5f} ({max_value[0]})')
        
    return results_list

def load_from_LLMPruner(model_path):
    # <LLM-Pruner: On the Structural Pruning of Large Language Models>
    # input format: f"/usr/workdir/LLM-Pruner/prune_log/{model_name.replace('-', '_')}:{task_name}_prune_0.60/pytorch_model.bin"
    assert model_path.endswith('.bin')
    from transformers.utils import HF_MODULES_CACHE, TRANSFORMERS_DYNAMIC_MODULE_NAME
    sys.path.append('/usr/workdir/LLM-Pruner')
    sys.path.append(HF_MODULES_CACHE)
    import transformers_modules

    print("Load from Pruned Model: {}".format(model_path))
    pruned_dict = torch.load(model_path, map_location='cpu')
    tokenizer = pruned_dict['tokenizer']
    model = pruned_dict['model'].half()
    return model, tokenizer

def load_from_ensemble(model_path, moe_args, mix_args):
    assert model_path.endswith('ensemble')
    from ensemble import post_process_config
    sys.path.append("/usr/workdir/MoEfication/models")
    from ensemble_llama import LlamaForCausalLMEnsemble
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    assert moe_args is not None
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    post_process_config(config, moe_args, mix_args, lora_rank=8, lora_alpha=16)
    if 'llama' in model_path:
        model = LlamaForCausalLMEnsemble.from_pretrained(model_path, config=config, torch_dtype=torch.float16)
    elif 'chatglm' in model_path:
        model = AutoModelForCausalLM.from_pretrained(model_path, config=config, torch_dtype=torch.float16, trust_remote_code=True)
    return model, tokenizer

def lora_from_base(model_path, init):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if init:
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
    else:
        model = AutoModelForCausalLM(AutoConfig.from_pretrained(model_path, trust_remote_code=True)).half()
    return model, tokenizer

def get_model_tokenizer(model_path, init = True, moe_args = None, mix_args = None):
    if os.path.isdir(model_path):
        if model_path.endswith('ensemble'):
            model, tokenizer = load_from_LLMPruner(model_path, moe_args, mix_args)
        else:    
            model, tokenizer = lora_from_base(model_path, init)
                
        if hasattr(model.config, 'max_position_embeddings'):
            model.seqlen = model.config.max_position_embeddings
    elif os.path.isfile(model_path):
        model, tokenizer = load_from_LLMPruner(model_path)
        
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.convert_ids_to_tokens(0)
        
    return model, tokenizer


def main():
    model_name = 'llama3.2-1b'  # ['ReluLLaMA-2-7B', 'llama-7b', 'llama2-7b', 't5-base', 'opt-6.7b', 'chatglm3-6b', 'relu-chatglm3-6b', 'llama3.2-1b', 'prosparse-llama-2-7b']
    model_path = f'/usr/workdir/models/{model_name}'
    model, tokenizer = get_model_tokenizer(model_path)
    apply_chat_template = False
    num_fewshot = 0
    
    eval_ppls(model, tokenizer)
    
    task_names = ['arc_easy', 'arc_challenge', 'piqa', 'openbookqa', 'winogrande', 'sciq', 'siqa']
    eval_task_name = [Train2Eval.get(task_name, task_name) for task_name in task_names]
    write_out = False
    
    result_ori = eval_on_tasks(
        model=model, 
        tokenizer=tokenizer, 
        task_name=eval_task_name, 
        write_out=write_out, 
        model_name=model_name, 
        num_fewshot=num_fewshot,
        apply_chat_template=apply_chat_template,
    )
        
    print(f'original model:')
    for key, value in result_ori.items():
        print(f'{key}: \t{get_metric_result(value):.5f}')
    
    
if __name__ == '__main__':
    main()