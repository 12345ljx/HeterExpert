import random
import numpy as np
from scipy.optimize import minimize
from gurobipy import Model, GRB, quicksum

random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)

NUM_DOMAIN = 8
# DFF_HIDDEN_SIZE = 8192
NUM_MODULES = 512
NUM_HIDDEN_LAYERS = 16
NUM_EXPERT = 16
NUM_EXPERT_ACT = 8

def gurobi_solver():
    domains_data = np.load('./score5000/importance_score_reduced.npy')  # [num_layers, num_neurons(512), num_domains]
    for layer_idx in range(NUM_HIDDEN_LAYERS):
        score = domains_data[layer_idx]
        
        model = Model("Solver")
        x = model.addVars(NUM_MODULES, NUM_EXPERT, vtype=GRB.BINARY, name="x")
        y = model.addVars(NUM_EXPERT, NUM_DOMAIN, vtype=GRB.BINARY, name="y")
        
        model.setObjective(
            quicksum(y[expert_idx, domain_idx] * quicksum(score[neuron_idx, domain_idx] * x[neuron_idx, expert_idx] for neuron_idx in range(NUM_MODULES)) 
                    for expert_idx in range(NUM_EXPERT) 
                    for domain_idx in range(NUM_DOMAIN)),
            GRB.MAXIMIZE
            )
        
        sparsity = NUM_EXPERT_ACT / NUM_EXPERT
        
        model.addConstrs((quicksum(x[neuron_idx, expert_idx] for expert_idx in range(NUM_EXPERT)) == 1 for neuron_idx in range(NUM_MODULES)), "NeuronPlacement")
        model.addConstrs((quicksum(x[neuron_idx, expert_idx] for neuron_idx in range(NUM_MODULES)) >= 16 for expert_idx in range(NUM_EXPERT)), "MinimumExpert")
        model.addConstrs((quicksum(y[expert_idx, domain_idx] * quicksum(x[neuron_idx, expert_idx] for neuron_idx in range(NUM_MODULES))
                                for expert_idx in range(NUM_EXPERT)) <= sparsity * NUM_MODULES
                        for domain_idx in range(NUM_DOMAIN)), "SparseActivation")
        
        model.Params.MIPGap = 0.005
        model.optimize()
        print(f'Objective: {model.objval}')
        
        if model.status == GRB.OPTIMAL:
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
            np.savez(f'./model_split/results/neuron_grouping.layer{layer_idx}.npz', placement=placement, chosen_experts=chosen_experts)
        else:
            print("No optimal solution found.")
    
    
if __name__ == "__main__":
    gurobi_solver()