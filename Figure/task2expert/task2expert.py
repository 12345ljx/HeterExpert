import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.switch_backend('agg')
plt.rcParams['pdf.use14corefonts'] = True
import sys
sys.path.append('/usr/workdir/MoEfication/moefication')
from gates_analysis.gate_analyse_dynamic import dynamic_experts_weights
from tools.path_lib import get_gate_path
from moefication import MoEArgs

plt.switch_backend('agg')
plt.rcParams['pdf.use14corefonts'] = True
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica']  # 'Arial' 'Helvetica'

def get_data(layer_name='3'):
    moeargs = MoEArgs(
        model_name='llama3.2-1b',
        function_name=None,   # XXX
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
    task_list = ['arc_easy', 'arc_challenge', 'piqa', 'openbookqa', 'winogrande', 'sciq', 'siqa']
    task_activation = np.zeros((len(task_list), 8))
    for id, task_name in enumerate(task_list):
        gate_path = get_gate_path(task_name=task_name, **moeargs.to_dict())
        selected_experts = dynamic_experts_weights(gate_path, False)
        selected_weight_average = {layer: value['experts_weight'] / value['num_tokens'] for layer, value in selected_experts.items()}
        task_activation[id] += selected_weight_average[layer_name]
        
    return task_activation

def get_random_data():
    return np.random.rand(7, 8)

def plot_heatmap(data):
    fig, ax = plt.subplots(figsize=(11, 8))
    sns.heatmap(data, annot=False, fmt=".2f", cmap='Blues', linewidths=.5, ax=ax)

    # plt.title('mixGate_heatmap', fontsize=18, fontweight='bold')
    plt.xlabel('Experts', fontsize=23, labelpad=10, fontweight='bold')
    plt.ylabel('Tasks', fontsize=23, labelpad=10, fontweight='bold')
    
    ax.set_xticklabels([f'E{i}' for i in range(8)], rotation=0, fontsize=20, ha='center')
    ax.set_yticklabels(['ARC-e', 'ARC-c', 'PIQA', 'OBQA', 'WG', 'SciQ', 'SIQA'], rotation=0, fontsize=20)
    
    cbar = ax.collections[0].colorbar
    # cbar.set_label('Expert Count', fontsize=14, labelpad=10)
    cbar.ax.tick_params(labelsize=23)

    plt.subplots_adjust(left=0.15, bottom=0.13, right=0.99, top=0.97)
    plt.savefig("/usr/workdir/HeterExpert/Figure/task2expert/task2expert.pdf", format='pdf')
    plt.close()

def main():
    data = get_data()
    # data = get_random_data()
    plot_heatmap(data)
    
if __name__ == '__main__':
    main()
    