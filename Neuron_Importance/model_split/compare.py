import random
import numpy as np

NUM_DOMAIN = 8
DFF_HIDDEN_SIZE = 512
NUM_EXPERT = 16
NUM_EXPERT_ACT = 8
NUM_HIDDEN_LAYERS = 16


def gurobi(neurons_score, layer_idx):
    data = np.load(f'/usr/workdir/HeterExpert/Neuron_Importance/model_split/results/neuron_grouping.layer{layer_idx}.npz')
    placement = data['placement']   # [num_neurons,]
    chosen_experts = data['chosen_experts']  # [num_domains, num_experts]
    
    neurons_chosens = [np.where(placement == expert_idx)[0] for expert_idx in range(NUM_EXPERT)]
    experts_size = np.array([len(neurons_chosen) for neurons_chosen in neurons_chosens], dtype=np.int32)
    experts_score = np.array([np.sum(neurons_score[neurons_chosen], axis=0) for neurons_chosen in neurons_chosens])
        
    scores = 0
    for domain_idx in range(NUM_DOMAIN):
        domain_score = np.sum(chosen_experts[domain_idx] * experts_score[:, domain_idx])
        scores += domain_score
        
    print(f"Total score: {scores}")
    print(f"Experts size: {experts_size}")
        
def random_dp(neurons_score):
    placement = np.random.randint(0, NUM_EXPERT, size=DFF_HIDDEN_SIZE)
        
    neurons_chosens = [np.where(placement == expert_idx)[0] for expert_idx in range(NUM_EXPERT)]
    experts_size = np.array([len(neurons_chosen) for neurons_chosen in neurons_chosens], dtype=np.int32)
    experts_score = np.array([np.sum(neurons_score[neurons_chosen], axis=0) for neurons_chosen in neurons_chosens])
    
    scores_all = 0
    for domain_idx in range(NUM_DOMAIN):
        capacity = int(DFF_HIDDEN_SIZE * NUM_EXPERT_ACT / NUM_EXPERT)
        scores = np.zeros((NUM_EXPERT + 1, capacity + 1))
        for i in range(1, NUM_EXPERT + 1):
            for j in range(0, capacity + 1):
                scores[i][j] = scores[i - 1][j]
                if j >= experts_size[i - 1]:
                    scores[i][j] = max(scores[i][j], scores[i - 1][j - experts_size[i - 1]] + experts_score[i - 1][domain_idx])
        scores_all += scores[NUM_EXPERT][capacity]
            
    print(f"Total score: {scores_all}")
    print(f"Experts size: {experts_size}")
    

def main():
    domains_data = np.load('/usr/workdir/HeterExpert/Neuron_Importance/score5000/importance_score_reduced.npy')  # [num_layers, num_neurons(512), num_domains]
    for layer_idx in range(NUM_HIDDEN_LAYERS):
        print('-'*15, f'layer={layer_idx}', '-'*15)
        neurons_score = domains_data[layer_idx]
        gurobi(neurons_score, layer_idx)
        random_dp(neurons_score)

if __name__ == '__main__':
    main()