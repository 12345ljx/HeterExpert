import os
import random
import numpy as np
from scipy.optimize import minimize
from gurobipy import Model, GRB, quicksum

# NUM_MODULES = 512
NUM_MODULES = 128

NUM_DOMAIN = 8
NUM_HIDDEN_LAYERS = 16
NUM_EXPERT = 8
NUM_EXPERT_ACT = 4

def gurobi_solver():
    output_path = f'./Split/ilp_split/raw_data/llama3.2-1b/domains(task_type)/k{NUM_EXPERT_ACT}n{NUM_EXPERT}m{NUM_MODULES}'
    os.makedirs(output_path, exist_ok=True)
    
    domains_data = np.load(f'./Neuron_Importance/score/task_type/importance_score_reduced_{NUM_MODULES}.npz')['domains_data_reduced']  # [num_layers, num_neurons(512), num_domains]
    for layer_idx in range(NUM_HIDDEN_LAYERS):
        score = domains_data[layer_idx]
        
        model = Model("Solver")
        x = model.addVars(NUM_MODULES, NUM_EXPERT, vtype=GRB.BINARY, name="x")
        y = model.addVars(NUM_EXPERT, NUM_DOMAIN, vtype=GRB.BINARY, name="y")
        
        def get_var(expert_idx):
            s = [quicksum(score[neuron_idx, domain_idx] * x[neuron_idx, expert_idx] for neuron_idx in range(NUM_MODULES))
                       for domain_idx in range(NUM_DOMAIN)]
            mean = quicksum(s) / NUM_DOMAIN
            variance = quicksum((a - mean) ** 2 for a in s)
            return variance
        
        gamma = 1.0
        
        model.setObjective(
            quicksum(y[expert_idx, domain_idx] * quicksum(score[neuron_idx, domain_idx] * x[neuron_idx, expert_idx] for neuron_idx in range(NUM_MODULES))
                    for expert_idx in range(NUM_EXPERT) 
                    for domain_idx in range(NUM_DOMAIN))
            # + quicksum(get_var(expert_idx) for expert_idx in range(NUM_EXPERT)) # regularization
            # - gamma * quicksum(y[expert_idx, domain_idx] for expert_idx in range(NUM_EXPERT) for domain_idx in range(NUM_DOMAIN)) # modularity
            ,
            GRB.MAXIMIZE
            )
        
        sparsity = NUM_EXPERT_ACT / NUM_EXPERT
        min_expert_size = NUM_MODULES // (NUM_EXPERT * 2)
        max_expert_size = (NUM_MODULES * 2) // NUM_EXPERT
        
        assert NUM_EXPERT >= NUM_DOMAIN
        max_expert_num = (NUM_EXPERT // NUM_DOMAIN) * 2
        
        model.addConstrs((quicksum(x[neuron_idx, expert_idx] for expert_idx in range(NUM_EXPERT)) == 1 for neuron_idx in range(NUM_MODULES)), "NeuronPlacement")
        # model.addConstrs((quicksum(y[expert_idx, domain_idx] * quicksum(x[neuron_idx, expert_idx] for neuron_idx in range(NUM_MODULES))
        #                         for expert_idx in range(NUM_EXPERT)) <= sparsity * NUM_MODULES
        #                 for domain_idx in range(NUM_DOMAIN)), "SparseActivation")
        model.addConstrs((quicksum(y[expert_idx, domain_idx] for domain_idx in range(NUM_DOMAIN)) >= 1 for expert_idx in range(NUM_EXPERT)), "Anti-Collapse")
        model.addConstrs((quicksum(x[neuron_idx, expert_idx] for neuron_idx in range(NUM_MODULES)) >= min_expert_size for expert_idx in range(NUM_EXPERT)), "MinimumExpert")
        # model.addConstrs((quicksum(y[expert_idx, domain_idx] for expert_idx in range(NUM_EXPERT)) <= NUM_EXPERT_ACT for domain_idx in range(NUM_DOMAIN)), "Coherence")
        
        model.addConstrs((quicksum(y[expert_idx, domain_idx] for expert_idx in range(NUM_EXPERT)) <= max_expert_num for domain_idx in range(NUM_DOMAIN)), "Coherence")
        model.addConstrs((quicksum(x[neuron_idx, expert_idx] for neuron_idx in range(NUM_MODULES)) <= max_expert_size for expert_idx in range(NUM_EXPERT)), "MaximumExpert")
        
        # model.Params.MIPGap = 0.005
        model.Params.TimeLimit = 300
        try:
            model.optimize()
        except Exception as e:
            print(e)
        print(f'Objective: {model.objval}')
        
        if model.status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
            print("Optimal solution found.")
            placement = np.zeros(NUM_MODULES) - 1
            chosen_experts = np.zeros((NUM_DOMAIN, NUM_EXPERT))
            
            for neuron_idx in range(NUM_MODULES):
                for expert_idx in range(NUM_EXPERT):
                    if x[neuron_idx, expert_idx].X > 0.5:  # if x[i, j] is chosen
                        assert placement[neuron_idx] == -1
                        placement[neuron_idx] = expert_idx
                        # print(f"Neuron {neuron_idx} is placed in Expert {expert_idx}.")
            for expert_idx in range(NUM_EXPERT):
                for domain_idx in range(NUM_DOMAIN):
                    if y[expert_idx, domain_idx].X > 0.5:  # if y[j, t] is chosen
                        chosen_experts[domain_idx][expert_idx] = 1
                        # print(f"Box {expert_idx} is selected for metric {domain_idx}.")
        
            assert np.sum(placement == -1) == 0
            np.savez(f'{output_path}/neuron_grouping.layer{layer_idx}.npz', placement=placement, chosen_experts=chosen_experts)
        else:
            print("No optimal solution found.")
    
    
if __name__ == "__main__":
    random_seed = 42
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    gurobi_solver()