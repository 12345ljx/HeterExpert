# Simulated Annealing

import random
import math
import numpy as np

NUM_MODULES = 128

NUM_DOMAIN = 8
NUM_HIDDEN_LAYERS = 16
NUM_EXPERT = 8
NUM_EXPERT_ACT = 4

def init_solution():
    placement = np.random.randint(0, NUM_EXPERT, size=NUM_MODULES)
    return placement

def get_neighborhood(solution):
    new_solution = solution.copy()
    neuron_idx = random.sample(range(NUM_MODULES), 1)
    new_solution[neuron_idx] = random.sample(range(NUM_EXPERT), 1)
    return new_solution

def fitness(placement, scores):
    solution = [[] for _ in range(NUM_EXPERT)]
    for i, expert in enumerate(placement):
        solution[expert].append(i)
    
    expert_scores = np.zeros((NUM_EXPERT, NUM_DOMAIN))
    for i in range(NUM_EXPERT):
        tmp_score = scores[np.array(solution[i])]
        expert_scores[i] = tmp_score.sum(axis=0)  # [num_domains]
        
    total_score = 0
    for task in range(NUM_DOMAIN):
        experts_score = expert_scores[:, task]
        task_score = np.sum(np.partition(experts_score, -NUM_EXPERT_ACT)[-NUM_EXPERT_ACT:])
        total_score += task_score
    return total_score

def simulated_annealing(scores, initial_temp=1000, final_temp=1, alpha=0.995, max_iter=1000):
    current_solution = init_solution()
    current_fitness = fitness(current_solution, scores)
    best_solution = current_solution
    best_fitness = current_fitness
    temp = initial_temp
    
    for iteration in range(max_iter):
        print('best_fitness', best_fitness) 
        new_solution = get_neighborhood(current_solution)
        new_fitness = fitness(new_solution, scores)
        
        if new_fitness > current_fitness:
            current_solution = new_solution
            current_fitness = new_fitness
        else:
            probability = math.exp((new_fitness - current_fitness) / temp)
            if random.random() < probability:
                current_solution = new_solution
                current_fitness = new_fitness
        
        temp *= alpha
        
        if current_fitness > best_fitness:
            best_solution = current_solution
            best_fitness = current_fitness
        
        if temp < final_temp:
            break
    
    return best_solution, best_fitness

random.seed(42)
np.random.seed(42)

domains_data = np.load(f'/usr/workdir/HeterExpert/Neuron_Importance/score5000/importance_score_reduced_{NUM_MODULES}.npz')['domains_data_reduced']  # [num_layers, num_neurons(512), num_domains]

# for layer_idx in range(NUM_HIDDEN_LAYERS):
layer_idx = 0
scores = domains_data[layer_idx]
best_placement, best_fitness = simulated_annealing(scores)

solution = [[] for _ in range(NUM_EXPERT)]
for i, expert in enumerate(best_placement):
    solution[expert].append(i)
        
print("Best Solution:", solution)
print("Best Fitness:", best_fitness)