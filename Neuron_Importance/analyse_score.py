import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import normalize


def read_score(num_layers, num_neurons, num_domains, data_path):
    domains = [f'domain{i}' for i in range(num_domains)]
    domains_data = np.zeros((num_layers, num_neurons, num_domains))
    for i, domain in enumerate(domains):
        importance_file = f'{data_path}/{domain}/importance_score.pkl'
        with open(importance_file, "rb") as file:
            data = np.array(pickle.load(file))
            print(domain, f'max_value: {np.max(data)}', f'min_value: {np.min(data):.2f}', data.shape)
            domains_data[:, :, i] = data
    return domains_data

def plot_distribution(domains_data, layer_idx):
    num_domains = domains_data.shape[2]
    plt.figure(figsize=(8, 6))
    colors = plt.cm.plasma(np.linspace(0, 1, num_domains))
    for i in range(num_domains):
        data = domains_data[layer_idx, :, i]
        # filtered_data = data[data < np.mean(data) + 1 * np.std(data)]  # XXX
        sns.kdeplot(data, color=colors[i], fill=True, alpha=0.5, label=f'domain{i}')
    plt.title('Score Distribution')
    plt.xlabel('score')
    plt.ylabel('density')
    plt.legend()

    plt.savefig(f"/usr/workdir/HeterExpert/Neuron_Importance/distribution.pdf", format='pdf')
    plt.close()
    
def plot_neuron_score(domains_data):
    num_layers = domains_data.shape[0]
    num_neurons = domains_data.shape[1]
    num_domains = domains_data.shape[2]
    
    def get_neuron_scores(layer_idx, neuron_idx):
        """get the scores of a neuron over all domains"""
        values = domains_data[layer_idx][neuron_idx][:]
        values = np.concatenate((values, [values[0]]))  # DOMAIN_NUM + 1
        return values

    angles = np.linspace(0, 2 * np.pi, num_domains, endpoint=False).tolist()
    angles += angles[:1]

    fig, axs = plt.subplots(4, 4, figsize=(24, 24), subplot_kw=dict(polar=True))

    num_neurons_shown = 5
    colors = plt.cm.plasma(np.linspace(0, 1, num_neurons_shown))
    neurons_idx = random.sample(range(num_neurons), num_neurons_shown)
    
    for ax, layer_idx in zip(axs.flat, range(num_layers)):
        for i, neuron_idx in enumerate(neurons_idx):
            values = get_neuron_scores(layer_idx, neuron_idx)
            ax.fill(angles, values, color=colors[i], alpha=0.25)
            ax.plot(angles, values, color=colors[i], linewidth=2)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([f'domain{i}' for i in range(num_domains)])
        yticks = np.linspace(0, np.max(domains_data), 5)
        ytick_labels = [f'{int(t)}' for t in yticks] 
        # ax.set_yticks(yticks)
        # ax.set_yticklabels(ytick_labels, color="grey", size=12)
        # ax.legend()
        
    for ax in axs.flat:
        ax.set_ylim(0, 0.004)

    plt.title('importance score')
    plt.tight_layout()
    plt.savefig(f"/usr/workdir/HeterExpert/Neuron_Importance/neuron_score.pdf", format='pdf')
    plt.close()
    
def preprocess_score(domains_data):
    num_layers, num_neurons, num_domains = domains_data.shape
    domains_data = domains_data.reshape(-1, num_domains)
    # domains_data = np.log1p(domains_data)
    domains_data = normalize(domains_data, axis=0)
    domains_data = domains_data.reshape(num_layers, num_neurons, num_domains)
    return domains_data
    

def main():
    random_seed = 42
    random.seed(random_seed)
    
    DOMAIN_NUM = 8
    dff_hidden_size = 8192
    num_hidden_layers = 16

    data_path = '/usr/workdir/HeterExpert/Neuron_Importance/score5000'
    domains_data = read_score(num_hidden_layers, dff_hidden_size, DOMAIN_NUM, data_path)  # [num_layers, num_neurons, num_domains]
    domains_data = preprocess_score(domains_data)
    np.save(f'{data_path}/importance_score(non_log).npy', domains_data)
 
    plot_distribution(domains_data, layer_idx=0)
    plot_neuron_score(domains_data)
    

if __name__ == '__main__':
    main()