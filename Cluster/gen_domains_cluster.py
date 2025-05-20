import os
import random
import numpy as np
from pathlib import Path

import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from datasets import concatenate_datasets
from datasets.io.json import JsonDatasetReader
from tqdm import tqdm
from pathlib import Path, PosixPath
import matplotlib.pyplot as plt

def read_examples(task_path: PosixPath):
    json_file = task_path / 'train.json'
    if not json_file.exists():
        raise FileExistsError(f'{json_file.name} doesn\'t have the train file')
    dataset = JsonDatasetReader(str(json_file), split='train').read()
    return dataset

def load_embedding_model():
    embedding_model = SentenceTransformer('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True, device='cuda:0')
    return embedding_model

def get_tasks_embeddings(sample_count, random_seed, matryoshka_dim=None, write_file=False):
    cache_path = './data/embeddings.npz'
    if os.path.exists(cache_path) and not write_file:
        load_data = np.load(cache_path)
        return load_data['embeddings'], load_data['labels']
    
    directory = Path('./data')
    subdirs = [d for d in directory.iterdir() if d.is_dir()]
    embedding_model = load_embedding_model()
    embeddings_all, labels_all = None, None
    for task_path in tqdm(subdirs):
        task_name = task_path.name.replace('_template', '')
        if task_name == 'mix_data': continue
        dataset = read_examples(task_path)
        sample_count_task = min(sample_count, len(dataset))
        dataset = dataset.shuffle(seed=random_seed).select(range(sample_count_task))
        
        sentences = ['clustering: {}\n{}'.format(item['instruction'], item['output']) for item in dataset]
        if matryoshka_dim is None:
            embeddings = embedding_model.encode(sentences)  # array(sample_count_task, 768)
        else:
            embeddings = embedding_model.encode(sentences, convert_to_tensor=True)
            embeddings = F.layer_norm(embeddings, normalized_shape=(embeddings.shape[1],))
            embeddings = embeddings[:, :matryoshka_dim]
            embeddings = embeddings.cpu().numpy()
        labels = np.array([task_name] * sample_count_task)
        
        if labels_all is None or embeddings_all is None:
            labels_all, embeddings_all = labels, embeddings
        else:
            labels_all = np.concatenate((labels_all, labels))
            embeddings_all = np.concatenate((embeddings_all, embeddings), axis=0)
            
    if write_file:
        np.savez(cache_path, embeddings=embeddings_all, labels=labels_all)
        
    return embeddings_all, labels_all
    
def merge_domain_data(domains_data):
    domain_count = []
    for data_sets in domains_data:
        domain_count.append(sum(data.num_rows for data in data_sets))
    
    domains_data_merge = [concatenate_datasets(domain) for domain in domains_data]
    
    if domain_count != [domain.num_rows for domain in domains_data_merge]:
        raise ValueError
    
    return domains_data_merge

def main():
    random_seed = 42
    random.seed(random_seed)
    
    domains_num = 8
    cluster_sample_count = 2000
    embeddings, _ = get_tasks_embeddings(cluster_sample_count, random_seed)
    kmeans = KMeans(n_clusters=domains_num, random_state=33)
    kmeans.fit_predict(normalize(embeddings))
    
    sample_count = 5000
    domains_data = [[] for _ in range(domains_num)]
    embedding_model = load_embedding_model()
    
    embeddings_all, tasks_labels_all, clusters_labels_all = None, None, None
    directory = Path('./Cluster/instruct_data')
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
    
    domains_data_merge = merge_domain_data(domains_data)
    print('Domain Examples Number:', [domain.num_rows for domain in domains_data_merge])
    for idx, domain in enumerate(domains_data_merge):
        domain_path = f'./Cluster/domains/cluster/domain{idx}'
        domain.save_to_disk(domain_path)
    
if __name__ == '__main__':
    main()