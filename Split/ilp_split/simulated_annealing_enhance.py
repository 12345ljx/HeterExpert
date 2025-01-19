# Simulated Annealing

import random
import math
import numpy as np

NUM_MODULES = 128

NUM_DOMAIN = 8
NUM_HIDDEN_LAYERS = 16
NUM_EXPERT = 8
NUM_EXPERT_ACT = 4

def get_neighborhood(placement):
    new_placement = placement.copy()
    neuron_a, neuron_b = random.sample(range(NUM_MODULES), 2)
    new_placement[neuron_a], new_placement[neuron_b] = new_placement[neuron_b], new_placement[neuron_a]
    return new_placement

def fitness(placement, chosen_experts, scores):
    neurons_chosens = [np.where(placement == expert_idx)[0] for expert_idx in range(NUM_EXPERT)]
    expert_scores = np.array([np.sum(scores[neurons_chosen], axis=0) for neurons_chosen in neurons_chosens])  # [num_experts, num_domains]

    total_score = 0
    for domain_idx in range(NUM_DOMAIN):
        total_score += np.sum(chosen_experts[domain_idx] * expert_scores[:, domain_idx])
    return total_score

def simulated_annealing(init_solution, scores, initial_temp=0.07, final_temp=0.0001, alpha=0.995, iter_max=500):
    current_placement, chosen_experts = init_solution['placement'], init_solution['chosen_experts']
    current_fitness = fitness(current_placement, chosen_experts, scores)
    best_placement, best_fitness = current_placement, current_fitness
    
    temp = initial_temp
    while temp >= final_temp:
        for iter in range(iter_max):
            # print('best_fitness', best_fitness) 
            new_placement = get_neighborhood(current_placement)
            new_fitness = fitness(new_placement, chosen_experts, scores)
            
            if new_fitness > current_fitness:
                current_placement, current_fitness = new_placement, new_fitness
            else:
                probability = math.exp((new_fitness - current_fitness) / temp)
                if random.random() < probability:
                    current_placement, current_fitness = new_placement, new_fitness
            
            if current_fitness > best_fitness:
                best_placement, best_fitness = current_placement, current_fitness
        
        temp *= alpha
    
    return best_placement, best_fitness

def main():
    random.seed(42)
    np.random.seed(42)

    domains_data = np.load(f'/usr/workdir/HeterExpert/Neuron_Importance/score5000/importance_score_reduced_{NUM_MODULES}.npz')['domains_data_reduced']  # [num_layers, num_neurons(512), num_domains]
    answer_path = f'/usr/workdir/HeterExpert/Split/ilp_split/raw_data/llama3.2-1b/domains(module_stable)/k{NUM_EXPERT_ACT}n{NUM_EXPERT}m{NUM_MODULES}'
    
    for layer_idx in range(NUM_HIDDEN_LAYERS):
    # layer_idx = 0
        ilp_answer = np.load(f'{answer_path}/neuron_grouping.layer{layer_idx}.npz')
        init_fitness = fitness(ilp_answer['placement'], ilp_answer['chosen_experts'], domains_data[layer_idx])
        # print("Initial Fitness:", init_fitness)
        
        best_placement, best_fitness = simulated_annealing(ilp_answer, domains_data[layer_idx])

        neurons_chosens = [np.where(best_placement == expert_idx)[0] for expert_idx in range(NUM_EXPERT)]
        # print("Best Solution:", neurons_chosens)
        # print("Best Fitness:", best_fitness)
        if best_fitness > init_fitness:
            print("Improved!")
        else:
            print("Not improved!")

if __name__ == '__main__':
    main()