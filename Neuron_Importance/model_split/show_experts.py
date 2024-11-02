import numpy as np
import matplotlib.pyplot as plt


NUM_DOMAIN = 8
DFF_HIDDEN_SIZE = 512
NUM_EXPERT = 16
NUM_EXPERT_ACT = 8

def show_experts_power(experts_score, experts_size, layer_idx):
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
    
    plt.savefig(f"/usr/workdir/HeterExpert/Neuron_Importance/model_split/experts_power(random).layer{layer_idx}.pdf", format='pdf')
    plt.close()
    

def main():
    domains_data = np.load('/usr/workdir/HeterExpert/Neuron_Importance/score5000/importance_score_reduced.npy')  # [num_layers, num_neurons, num_domains]
    layer_idx = 15
    score = domains_data[layer_idx]
    
    data = np.load(f'/usr/workdir/HeterExpert/Neuron_Importance/model_split/results/neuron_grouping.layer{layer_idx}.npz')
    placement = data['placement']   # [num_neurons,]
    placement = np.random.randint(0, NUM_EXPERT, size=DFF_HIDDEN_SIZE)
    chosen_experts = data['chosen_experts']  # [num_domains, num_experts]

    experts_size = np.zeros(NUM_EXPERT, dtype=np.int32)
    experts_score = np.zeros((NUM_EXPERT, NUM_DOMAIN))
    for expert_idx in range(NUM_EXPERT):
        neurons_idx = np.where(placement == expert_idx)[0]
        experts_size[expert_idx] = len(neurons_idx)
        experts_score[expert_idx] = np.sum(score[neurons_idx], axis=0)
        
    show_experts_power(experts_score, experts_size, layer_idx)

if __name__ == '__main__':
    main()   