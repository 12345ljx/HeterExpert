import os
import random
import numpy as np
from pathlib import Path, PosixPath
from datasets.io.json import JsonDatasetReader
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import normalize
from tqdm import tqdm

os.environ["http_proxy"] = "http://10.129.202.92:7900"
os.environ["https_proxy"] = os.environ["http_proxy"]

def read_examples(task_path: PosixPath):
    json_file = task_path / 'train.json'
    if not json_file.exists():
        raise FileExistsError(f'{json_file.name} doesn\'t have the train file')
    dataset = JsonDatasetReader(str(json_file), split='train').read()
    return dataset

def load_embedding_model():
    model_path = '/usr/workdir/MoEfication/models/nomic-embed-text-v1.5'
    embedding_model = SentenceTransformer(model_path, trust_remote_code=True, device='cuda:1')
    return embedding_model

def get_tasks_embeddings(sample_count, random_seed, write_file=False):
    directory = Path('/usr/workdir/HeterExpert/data')
    subdirs = [d for d in directory.iterdir() if d.is_dir()]
    
    embedding_model = load_embedding_model()
    
    embeddings_all, labels_all = None, None
    for task_path in tqdm(subdirs):
        task_name = task_path.name.replace('_template', '')
        dataset = read_examples(task_path)
        sample_count_task = min(sample_count, len(dataset))
        dataset = dataset.shuffle(seed=random_seed).select(range(sample_count_task))
        
        sentences = ['clustering: {}\n{}'.format(item['instruction'], item['output']) for item in dataset]
        embeddings = embedding_model.encode(sentences)  # array(sample_count_task, 768)
        labels = np.array([task_name] * sample_count_task)
        
        if labels_all is None or embeddings_all is None:
            labels_all, embeddings_all = labels, embeddings
        else:
            labels_all = np.concatenate((labels_all, labels))
            embeddings_all = np.concatenate((embeddings_all, embeddings), axis=0)
            
    if write_file:
        np.savez('/usr/workdir/HeterExpert/data/embeddings.npz', embeddings=embeddings_all, labels=labels_all)
        
    return embeddings_all, labels_all

def cluster_embeddings(embeddings, domains_num, random_seed):
    kmeans = KMeans(n_clusters=domains_num, random_state=random_seed)
    clusters = kmeans.fit_predict(embeddings)
    
    print(f"Cluster Number {domains_num}: ", end='')
    print([np.sum(clusters == i) for i in range(domains_num)])
    
    return clusters, kmeans.cluster_centers_, kmeans.inertia_

def dim_reduction(embeddings, centers, method, random_seed):
    if method == 'pca':
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings)
        centers_2d = pca.transform(centers)
    elif method == 'tsne':
        tsne = TSNE(n_components=2, perplexity=50, random_state=random_seed, n_iter=5000)
        embeddings_centers = np.concatenate((embeddings, centers), axis=0)
        embeddings_centers_2d = tsne.fit_transform(embeddings_centers)
        embeddings_2d, centers_2d = embeddings_centers_2d[:-centers.shape[0]], embeddings_centers_2d[-centers.shape[0]:]
    else:
        raise ValueError('The method should be either pca or tsne')
    
    return embeddings_2d, centers_2d

def plot(embeddings_2d, labels, centers_2d, show_count, figure_path):
    fig, axs = plt.subplots(4, 7, figsize=(20, 12))
    unique_labels = np.unique(labels)
    colors = plt.cm.plasma(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        ax = axs[i // 7][i % 7]
        data_task = embeddings_2d[labels == label]
        data_task = data_task[:show_count]
        ax.scatter(data_task[:, 0], data_task[:, 1], color=colors[i], label=label, alpha=0.7)
        ax.scatter(centers_2d[:, 0], centers_2d[:, 1], color='black', marker='X', s=50) 
        ax.set_title(f'Distribution of Type {label}')
        ax.set_xlabel('PCA Component 1')  # 'TSNE Component 1'
        ax.set_ylabel('PCA Component 2')  # 'TSNE Component 2'
        ax.legend()
    
    x_min, x_max = embeddings_2d[:, 0].min(), embeddings_2d[:, 0].max()
    y_min, y_max = embeddings_2d[:, 1].min(), embeddings_2d[:, 1].max()
    for line in axs:
        for ax in line:
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
    
    plt.tight_layout()
    plt.savefig(figure_path, format='pdf')
    plt.close()

def main():
    random_seed = 42
    random.seed(random_seed)
    
    sample_count = 2000
    show_count = 500
    assert sample_count >= show_count
    
    # silhouette_score achieves high score when domains_num=8 && cluster random_seed=33
    domains_num = 8
    
    embeddings, labels = get_tasks_embeddings(sample_count, random_seed)
    embeddings = normalize(embeddings)
    clusters, centers, inertia = cluster_embeddings(embeddings, domains_num, random_seed=33)
    embeddings_2d, centers_2d = dim_reduction(embeddings, centers, 'pca', random_seed)
    
    figure_path = "/usr/workdir/HeterExpert/PCA_sub.pdf"
    plot(embeddings_2d, labels, centers_2d, show_count, figure_path)
    
def determine_cluster_num():
    random_seed = 42
    random.seed(random_seed)
    
    sample_count = 2000
    embeddings, _ = get_tasks_embeddings(sample_count, random_seed)
    embeddings = normalize(embeddings)
    
    domains_num_range = range(6, 16, 2)
    random_seed_range = range(33, 50, 3)
    
    clusters_dict = {random_seed: {} for random_seed in random_seed_range}
    inertia_dict = {random_seed: {} for random_seed in random_seed_range}
    for random_seed in random_seed_range:
        for domains_num in domains_num_range:
            clusters, _, inertia = cluster_embeddings(embeddings, domains_num, random_seed)
            clusters_dict[random_seed][domains_num] = clusters
            inertia_dict[random_seed][domains_num] = inertia
            
    def evaluate_clusters(metric):
        for random_seed in random_seed_range:
            scores = []
            for domains_num in domains_num_range:
                score = metric(embeddings, clusters_dict[random_seed][domains_num])
                scores.append(score)
                # print(f"Cluster Seed {random_seed} Number {domains_num} : {score}")
            plt.plot(domains_num_range, scores, label=f"Seed {random_seed}")
        
        plt.title(f'{metric.__name__}')
        plt.xlabel('Number of clusters (k)')
        plt.ylabel(f'{metric.__name__}')
        plt.legend()
        plt.savefig(f"/usr/workdir/HeterExpert/cluster_evaluate/{metric.__name__}.pdf", format='pdf')
        plt.close()
        
    for metric in [silhouette_score, calinski_harabasz_score, davies_bouldin_score]:
        evaluate_clusters(metric)
        
    for random_seed in random_seed_range:
        scores = inertia_dict[random_seed].values()
        plt.plot(domains_num_range, scores, label=f"Seed {random_seed}")
    
    plt.title('Inertia')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('inertia')
    plt.legend()
    plt.savefig(f"/usr/workdir/HeterExpert/cluster_evaluate/inertia.pdf", format='pdf')
    plt.close()
    

if __name__ == '__main__':
    # main()
    determine_cluster_num()
    