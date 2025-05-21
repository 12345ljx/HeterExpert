import pickle
import random
import numpy as np
from sklearn.preprocessing import normalize

def read_score(num_layers, num_neurons, num_domains, data_path):
    if 'task_single' in data_path:
        domains = ['siqa', 'boolq', 'anli', 'hellaswag', 'gsm8k', 'sst2', 'cb', 'winogrande']
    else:
        domains = [f'domain{i}' for i in range(num_domains)]
        
    domains_data = np.zeros((num_layers, num_neurons, num_domains))
    for i, domain in enumerate(domains):
        importance_file = f'{data_path}/{domain}/importance_score.pkl'
        with open(importance_file, "rb") as file:
            data = np.array(pickle.load(file))
            print(domain, f'max_value: {np.max(data)}', f'min_value: {np.min(data):.2f}', data.shape)
            domains_data[:, :, i] = data
            
    return domains_data
    
def preprocess_score(domains_data):
    num_layers, num_neurons, num_domains = domains_data.shape
    domains_data = domains_data.reshape(-1, num_domains)
    domains_data = np.log1p(domains_data)
    domains_data = normalize(domains_data, axis=0)
    domains_data = domains_data.reshape(num_layers, num_neurons, num_domains)
    return domains_data
    

def main():
    random_seed = 42
    random.seed(random_seed)
    
    DOMAIN_NUM = 8
    dff_hidden_size, num_hidden_layers = 8192, 16  # llama3.2-1b
    # dff_hidden_size, num_hidden_layers = 8192, 28  # llama3.2-3b

    data_path = './Neuron_Score/score/cluster/llama3.2-1b'
    domains_data = read_score(num_hidden_layers, dff_hidden_size, DOMAIN_NUM, data_path)  # [num_layers, num_neurons, num_domains]
    domains_data = preprocess_score(domains_data)
    print(domains_data.min(), domains_data.max())
    np.save(f'{data_path}/importance_score.npy', domains_data)
    

if __name__ == '__main__':
    main()