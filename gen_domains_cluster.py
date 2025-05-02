import random
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from datasets import concatenate_datasets
from tqdm import tqdm
import matplotlib.pyplot as plt

from cluster_plot import get_tasks_embeddings, read_examples, load_embedding_model, dim_reduction

def show_domains(domains_compose: np.ndarray, tasks_list: list):
    for task_idx, task in enumerate(tasks_list):
        print(f"Task {task}:", domains_compose[:, task_idx])
        
    domains_sum = np.sum(domains_compose, axis=1)
    print(f'Domain Examples Number: {domains_sum}')
    
    domains_rate = np.divide(domains_compose, domains_sum.reshape(-1, 1)) * 100
    print(' '.join(tasks_list))
    for domain_idx, rate in enumerate(domains_rate):
        print(f'Domain {domain_idx}:', ' '.join([f'{percent:.1f}%' for percent in rate]))

def analyse_cluster(kmeans: KMeans, sample_count: int, random_seed: int):
    embeddings, labels = get_tasks_embeddings(sample_count, random_seed)
    clusters = kmeans.predict(normalize(embeddings))
    
    unique_labels = np.unique(labels)
    domains_compose = np.zeros((kmeans.n_clusters, len(unique_labels)), dtype=np.int32)
    for task_idx, label in enumerate(unique_labels):
        clusters_task = clusters[labels == label]
        for domain in range(kmeans.n_clusters):
            domains_compose[domain][task_idx] += np.sum(clusters_task == domain)
        
    np.save('/usr/workdir/HeterExpert/domains/cluster/domains_compose.npy', domains_compose)
    show_domains(domains_compose, unique_labels)
    
def merge_domain_data(domains_data):
    domain_count = []
    for data_sets in domains_data:
        domain_count.append(sum(data.num_rows for data in data_sets))
    
    domains_data_merge = [concatenate_datasets(domain) for domain in domains_data]
    
    if domain_count != [domain.num_rows for domain in domains_data_merge]:
        raise ValueError
    
    return domains_data_merge

def plot(embeddings_2d, tasks_labels, clusters_labels, centers_2d, figure_path):
    fig, axs = plt.subplots(1, 2, figsize=(25, 12))
    tasks_labels_unique = np.unique(tasks_labels)
    clusters_labels_unique = np.unique(clusters_labels)
    
    colors_task = plt.cm.plasma(np.linspace(0, 1, len(tasks_labels_unique))) 
    colors_cluster = plt.cm.viridis(np.linspace(0, 1, len(clusters_labels_unique))) 
    
    for i, tasks_label in enumerate(tasks_labels_unique):
        data_task = embeddings_2d[tasks_labels == tasks_label]
        axs[0].scatter(data_task[:, 0], data_task[:, 1], color=colors_task[i], label=tasks_label, alpha=0.7)
    
    for i, clusters_label in enumerate(clusters_labels_unique):
        data_domain = embeddings_2d[clusters_label == clusters_labels]
        axs[1].scatter(data_domain[:, 0], data_domain[:, 1], color=colors_cluster[i], label=clusters_label, alpha=0.7)
        
    x_min, x_max = embeddings_2d[:, 0].min(), embeddings_2d[:, 0].max()
    y_min, y_max = embeddings_2d[:, 1].min(), embeddings_2d[:, 1].max()
    for ax in axs.flat:
        ax.scatter(centers_2d[:, 0], centers_2d[:, 1], color='black', marker='X', s=50)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_title(f'Distribution of Tasks')
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(figure_path, format='pdf')
    plt.close()

def main():
    random_seed = 42
    random.seed(random_seed)
    
    domains_num = 8
    cluster_sample_count = 2000
    embeddings, _ = get_tasks_embeddings(cluster_sample_count, random_seed)
    kmeans = KMeans(n_clusters=domains_num, random_state=33)
    kmeans.fit_predict(normalize(embeddings))
    
    sample_count = 5000
    # analyse_cluster(kmeans, sample_count, random_seed)
    
    domains_data = [[] for _ in range(domains_num)]
    
    embedding_model = load_embedding_model()
    
    embeddings_all, tasks_labels_all, clusters_labels_all = None, None, None
    directory = Path('/usr/workdir/HeterExpert/data')
    subdirs = [d for d in directory.iterdir() if d.is_dir()]
    for task_path in tqdm(subdirs):
        task_name = task_path.name.replace('_template', '')
        if task_name == 'mix_data': continue
        dataset = read_examples(task_path)
        sample_count_task = min(sample_count, len(dataset))
        dataset = dataset.shuffle(seed=random_seed).select(range(sample_count_task))
        
        sentences = ['clustering: {}\n{}'.format(item['instruction'], item['output']) for item in dataset]
        embeddings = embedding_model.encode(sentences)  # array(sample_count_task, 768)
        embeddings = normalize(embeddings)
        clusters = kmeans.predict(embeddings)
        tasks_labels = np.array([task_name] * sample_count_task)
        
        if embeddings_all is None:
            embeddings_all, tasks_labels_all, clusters_labels_all = embeddings, tasks_labels, clusters
        else:
            embeddings_all = np.concatenate((embeddings_all, embeddings), axis=0)
            tasks_labels_all = np.concatenate((tasks_labels_all, tasks_labels))
            clusters_labels_all = np.concatenate((clusters_labels_all, clusters))
        
        for domain in range(domains_num):
            count = np.sum(clusters == domain)
            if count == 0: continue
            data_for_domain = dataset.select(np.where(clusters == domain)[0])
            assert data_for_domain.num_rows == count
            data_for_domain = data_for_domain.add_column("task", [task_name] * count)
            domains_data[domain].append(data_for_domain)
    
    # embeddings_2d, centers_2d = dim_reduction(embeddings_all, kmeans.cluster_centers_, 'tsne', random_seed)
    # figure_path = "/usr/workdir/HeterExpert/domains/cluster/tasks_distribution.pdf"
    # plot(embeddings_2d, tasks_labels_all, clusters_labels_all, centers_2d, figure_path)
    
    domains_data_merge = merge_domain_data(domains_data)
    print('Domain Examples Number:', [domain.num_rows for domain in domains_data_merge])
    for idx, domain in enumerate(domains_data_merge):
        domain_path = f'/usr/workdir/HeterExpert/domains/cluster/domain{idx}'
        domain.save_to_disk(domain_path)
    
            
if __name__ == '__main__':
    main()