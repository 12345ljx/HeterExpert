import os
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import sys
sys.path.append('/usr/workdir/balanced_k_means')
from clustering.equal_groups import EqualGroupsKMeans

os.environ["http_proxy"] = "http://10.129.202.92:7900"
os.environ["https_proxy"] = os.environ["http_proxy"]

def main():
    random_seed = 42
    random.seed(random_seed)
    
    sample_count = 2000
    show_count = 500
    assert sample_count >= show_count
    domains_num = 8
    
    load_data = np.load('/usr/workdir/HeterExpert/data/embeddings.npz')
    embeddings_all, labels_all = load_data['embeddings'], load_data['labels']
    print('embeddings_all shape:', embeddings_all.shape)
    print('labels_all shape:', labels_all.shape)
    
    pca = PCA(n_components=2)
    data_2d = pca.fit_transform(embeddings_all)
    
    clf = EqualGroupsKMeans(n_clusters=domains_num)
    clf.fit(embeddings_all)
    clusters = clf.labels_
    centers = clf.cluster_centers_

    # kmeans = KMeans(n_clusters=domains_num)
    # clusters = kmeans.fit_predict(embeddings_all)
    # centers = kmeans.cluster_centers_
    
    print('clusters shape:', clusters.shape)
    print('centers shape:', centers.shape)
    for i in range(domains_num):
        print("Cluster {}: {}".format(i, np.sum(clusters == i)))
    centers_2d = pca.transform(centers)
    
    fig, axs = plt.subplots(4, 7, figsize=(20, 12))
    
    unique_labels = np.unique(labels_all)
    colors = plt.cm.plasma(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        ax = axs[i // 7][i % 7]
        data_task = data_2d[labels_all == label]
        data_task = data_task[:show_count]
        ax.scatter(data_task[:, 0], data_task[:, 1], color=colors[i], label=label, alpha=0.7)
        ax.scatter(centers_2d[:, 0], centers_2d[:, 1], color='black', marker='X', s=50) 

        ax.set_title('Distribution of Type {}'.format(label))
        ax.set_xlabel('PCA Component 1')
        ax.set_ylabel('PCA Component 2')
        ax.legend()
    
    x_min, x_max = data_2d[:, 0].min(), data_2d[:, 0].max()
    y_min, y_max = data_2d[:, 1].min(), data_2d[:, 1].max()

    for line in axs:
        for ax in line:
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
    
    plt.tight_layout()
    plt.savefig("/usr/workdir/HeterExpert/PCA_sub(balanced).pdf", format='pdf')
    plt.close()

if __name__ == '__main__':
    main()
    