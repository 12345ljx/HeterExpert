import numpy as np
import matplotlib.pyplot as plt


NUM_DOMAIN = 8
DFF_HIDDEN_SIZE = 512
NUM_EXPERT = 16
NUM_EXPERT_ACT = 8

def show_experts_power(experts_score, experts_size, figure_path):
    labels = ['D{}'.format(i) for i in range(NUM_DOMAIN)]
    
    fig, axs = plt.subplots(4, 4, figsize=(15, 15))
    for ax, expert_idx in zip(axs.flat, range(experts_score.shape[0])):
        expert_score = experts_score[expert_idx]
        ax.bar(labels, expert_score)
        ax.set_title(f'Expert {expert_idx} (Neurons: {experts_size[expert_idx]})')
    
    plt.tight_layout()
    y_max = experts_score.max()
    for ax in axs.flat:
        ax.set_ylim(0, y_max)
    
    plt.savefig(figure_path, format='pdf')
    plt.close()
    
def get_placement(layer_idx, random_split):
    if random_split:
        placement = np.random.randint(0, NUM_EXPERT, size=DFF_HIDDEN_SIZE)
    else:
        data = np.load(f'/usr/workdir/HeterExpert/Split/ilp_split/raw_data/{NUM_EXPERT}/neuron_grouping.layer{layer_idx}.npz')
        placement = data['placement']   # [num_neurons,]
    return placement

def main():
    layer_idx = 5
    random_split = False
    placement = get_placement(layer_idx, random_split)
    
    domains_data = np.load(f'/usr/workdir/HeterExpert/Neuron_Importance/score5000/importance_score_reduced_{DFF_HIDDEN_SIZE}.npz')['domains_data_reduced']  # [num_layers, num_neurons, num_domains]
    score = domains_data[layer_idx]

    experts_score = np.zeros((NUM_EXPERT, NUM_DOMAIN))
    for expert_idx in range(NUM_EXPERT):
        neurons_idx = np.where(placement == expert_idx)[0]
        experts_score[expert_idx] = np.sum(score[neurons_idx], axis=0)
    
    experts_size = [len(np.where(placement == expert_idx)[0]) for expert_idx in range(NUM_EXPERT)]
    
    if random_split:
        figure_path = f"/usr/workdir/HeterExpert/Split/ilp_split/experts_power(random).layer{layer_idx}.pdf"
    else:
        figure_path = f"/usr/workdir/HeterExpert/Split/ilp_split/experts_power.layer{layer_idx}.pdf"
        
    show_experts_power(experts_score, experts_size, figure_path)

if __name__ == '__main__':
    main()   