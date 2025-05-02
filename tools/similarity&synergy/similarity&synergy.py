import os
import random
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

import seaborn as sns
import matplotlib.pyplot as plt
plt.switch_backend('agg')
plt.rcParams['pdf.use14corefonts'] = True

import sys
sys.path.append('/usr/workdir/MoEfication/moefication')
from moefication import moeficate, MoEArgs
from get_labels import get_labels_base_path, read_labels
from gates_analysis.gate_analyse_dynamic import load_data

sys.path.append('/usr/workdir/HeterExpert/tools/activated_params')
from activated_params import get_moe_model

from SimilarityObserve import SimilarityObserve

def plot_heatmap(data):
    fig, ax = plt.subplots(figsize=(12, 12))
    sns.heatmap(data, annot=False, fmt=".2f", cmap='Blues', linewidths=.5, ax=ax)

    # plt.title('mixGate_heatmap', fontsize=18, fontweight='bold')
    # plt.xlabel('Experts', fontsize=18, labelpad=10, fontweight='bold')
    # plt.ylabel('Tasks', fontsize=18, labelpad=10, fontweight='bold')
    
    ax.set_xticklabels([f'E{i}' for i in range(8)], rotation=0, fontsize=10, ha='center')
    ax.set_yticklabels([f'E{i}' for i in range(8)], rotation=0, fontsize=10)
    
    cbar = ax.collections[0].colorbar
    # cbar.set_label('Expert Count', fontsize=14, labelpad=10)
    cbar.ax.tick_params(labelsize=12)

    plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)
    plt.savefig("/usr/workdir/HeterExpert/tools/similarity&synergy/similarity.pdf", format='pdf')
    plt.close()

def main():
    model_name = 'llama3.2-1b'
    function_name = 'domains(module_stable)'
    task_name = 'mix_data'
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
    observer = SimilarityObserve()
    observe_function = observer.observe_library(model_name)
    
    sentences = load_data(task_name)
    # sentences = random.sample(sentences, min(5000, len(sentences)))
    sentences = sorted(sentences, key=len, reverse=True)
    model.eval().cuda()
    batch_size = 4
    
    res = np.zeros((model.config.num_hidden_layers, moeargs.num_expert, moeargs.num_expert))
    count = 0
    for index in tqdm(range(len(sentences) // batch_size + 1)) :
        batch_sentences = sentences[index * batch_size : (index + 1) * batch_size]
        if len(batch_sentences) == 0 :
            continue
        
        encoded_input = tokenizer(batch_sentences, return_tensors="pt", padding=True, truncation=True, max_length=1024)
        input_ids = encoded_input.input_ids.cuda()
        attention_mask = encoded_input.attention_mask.cuda()
        with torch.no_grad() :
            hooks = observe_function(model, res, attention_mask)
            model(input_ids=input_ids, attention_mask=attention_mask)
            observer.erase_hooks(hooks)
        count += 1
    res = res / count
    res = -res
    res = res[3]
    for i in range(8):
        res[i, i] = res.min()
    plot_heatmap(res)
    
    
    

if __name__ == '__main__':
    main()