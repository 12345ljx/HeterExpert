import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import Counter

plt.switch_backend('agg')
plt.rcParams['pdf.use14corefonts'] = True
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica']  # 'Arial' 'Helvetica'

import sys
sys.path.append('/usr/workdir/HeterExpert/Split/random_split')
from random_split_homo import averate_correction

NUM_DOMAIN = 8
# DFF_HIDDEN_SIZE = 512
DFF_HIDDEN_SIZE = 128
NUM_EXPERT = 8
NUM_EXPERT_ACT = 4

def show_experts_power_paper(experts_score, experts_size, figure_path):
    labels = ['D{}'.format(i) for i in range(NUM_DOMAIN)]
    
    selected_idx = [0, 1, 4]
    key_expert = [2, 0, 4]
    
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.8))
    for i, ax, expert_idx in zip(range(len(selected_idx)), axs.flat, selected_idx):
        expert_score = experts_score[expert_idx]
        bars = ax.bar(labels, expert_score, color='#c5daee')
        bars[key_expert[i]].set_color('#3383be')
        ax.set_title(f'E{expert_idx} (module size: {experts_size[expert_idx] * 64})', fontsize=18)
        
        y_max = expert_score.max()
        ax.set_ylim(y_max-0.35, y_max+0.15)
        ax.tick_params(axis='both', labelsize=15)
        
    plt.tight_layout()
    plt.savefig(figure_path, format='pdf')
    plt.close()


def show_experts_power(experts_score, experts_size, figure_path):
    labels = ['D{}'.format(i) for i in range(NUM_DOMAIN)]
    
    selected_idx = range(NUM_EXPERT)
    fig, axs = plt.subplots(2, 4, figsize=(12, 6))
    for ax, expert_idx in zip(axs.flat, selected_idx):
        expert_score = experts_score[expert_idx]
        ax.bar(labels, expert_score, color='#75b4d8')
        ax.set_title(f'E{expert_idx} (module size: {experts_size[expert_idx] * 64})', fontsize=18)
        
        y_max = expert_score.max()
        ax.set_ylim(y_max-0.35, y_max+0.15)
        # ax.set_ylim(1, y_max+0.15)
        ax.tick_params(axis='both', labelsize=15)
        
    # for ax, expert_idx in zip(axs.flat, range(experts_score.shape[0])):
    #     expert_score = experts_score[expert_idx]
    #     ax.bar(labels, expert_score)
    #     ax.set_title(f'Expert {expert_idx} (Neurons: {experts_size[expert_idx]})')
        
    #     y_max = expert_score.max()
    #     ax.set_ylim(y_max-0.35, y_max+0.15)
        
    
    plt.tight_layout()
    # y_max = experts_score.max()
    # for ax in axs.flat:
        # ax.set_ylim(0, y_max)
    
    plt.savefig(figure_path, format='pdf')
    plt.close()
    
def get_placement(model_name, layer_idx, random_split):
    if random_split:
        placement = np.random.randint(0, NUM_EXPERT, size=DFF_HIDDEN_SIZE)
        print(Counter(placement))
        average_num = DFF_HIDDEN_SIZE // NUM_EXPERT
        averate_correction({layer_idx: placement}, average_num)
        print(Counter(placement))
    else:
        data = np.load(f'/usr/workdir/HeterExpert/Split/ilp_split/raw_data/encode_error/{model_name}/domains(module_stable)/n{NUM_EXPERT}m{DFF_HIDDEN_SIZE}/neuron_grouping.layer{layer_idx}.npz')
        placement = data['placement']   # [num_neurons,]
    return placement

def main():
    model_name = 'llama3.2-1b'
    layer_idx = 0
    random_split = False
    placement = get_placement(model_name, layer_idx, random_split)
    
    domains_data = np.load(f'/usr/workdir/HeterExpert/Neuron_Importance/score/encode_error/cluster/{model_name}/importance_score_reduced_{DFF_HIDDEN_SIZE}.npz')['domains_data_reduced']  # [num_layers, num_neurons, num_domains]
    score = domains_data[layer_idx]

    experts_score = np.zeros((NUM_EXPERT, NUM_DOMAIN))
    for expert_idx in range(NUM_EXPERT):
        neurons_idx = np.where(placement == expert_idx)[0]
        experts_score[expert_idx] = np.sum(score[neurons_idx], axis=0)
    
    experts_size = [len(np.where(placement == expert_idx)[0]) for expert_idx in range(NUM_EXPERT)]
    
    output_path = "/usr/workdir/HeterExpert/Split/ilp_split/experts_power"
    os.makedirs(output_path, exist_ok=True)
    
    if random_split:
        figure_path = f"{output_path}/experts_power(random).layer{layer_idx}.pdf"
    else:
        figure_path = f"{output_path}/experts_power.layer{layer_idx}.pdf"
        
    show_experts_power_paper(experts_score, experts_size, figure_path)

if __name__ == '__main__':
    main()   