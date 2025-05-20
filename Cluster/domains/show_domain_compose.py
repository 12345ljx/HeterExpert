import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

import sys
sys.path.append('/usr/workdir/HeterExpert')
from cluster_plot import task_name_fmt

plt.switch_backend('agg')
plt.rcParams['pdf.use14corefonts'] = True
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica']  # 'Arial' 'Helvetica'

base_path = '/usr/workdir/HeterExpert/domains/cluster'
domains_compose = np.load(f'{base_path}/domains_compose.npy')
domains_num = 8

directory = Path('/usr/workdir/HeterExpert/data')
subdirs = [d for d in directory.iterdir() if d.is_dir()]
tasks_list = []
for task_path in subdirs:
    task_name = task_path.name.replace('_template', '')
    if task_name == 'mix_data': continue
    tasks_list.append(task_name)
labels = sorted(tasks_list)
tasks_num = len(labels)

def show_domains(domains_compose: np.ndarray, labels: list):
    colors = plt.cm.plasma(np.linspace(0, 1, tasks_num))
    fig, axs = plt.subplots(2, 4, figsize=(12, 5))
    for ax, domain in zip(axs.flat, range(domains_num)):
        domain_compose = domains_compose[domain]
        ax.pie(domain_compose, labels=None, colors=colors, autopct=None, startangle=140)
        ax.axis('equal')
        ax.set_title('Domain {}'.format(domain), fontsize=14)

    plt.subplots_adjust(right=0.65)

    handles = [plt.Line2D([0], [0], marker='o', color='w', label=task_name_fmt[label], markersize=10, markerfacecolor=color) 
            for label, color in zip(labels, colors)]
    fig.legend(handles=handles, loc='center right', bbox_to_anchor=(0.95, 0.5), 
            title="Tasks", ncol=2, fontsize=12, title_fontsize=14)
        
    # plt.tight_layout()

    plt.savefig(f"{base_path}/domains_compose.pdf", format='pdf')
    plt.close()
    
def show_tasks(domains_compose: np.ndarray, labels: list):
    tasks_compose = np.transpose(domains_compose)
    
    colors = plt.cm.plasma(np.linspace(0, 1, domains_num))
    fig, axs = plt.subplots(4, 7, figsize=(16, 10))
    for ax, task in zip(axs.flat, range(tasks_num)):
        task_compose = tasks_compose[task]
        ax.pie(task_compose, labels=None, colors=colors, autopct=None, startangle=140)
        ax.axis('equal')
        ax.set_title(f'{task_name_fmt[labels[task]]}', fontsize=14)
    
    for j in range(26, 28):
        fig.delaxes(axs.flatten()[j])

    plt.subplots_adjust(right=0.8)

    domains_label = [f"Domain {i}" for i in range(domains_num)]
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=label, markersize=10, markerfacecolor=color) 
            for label, color in zip(domains_label, colors)]
    fig.legend(handles=handles, loc='center right', bbox_to_anchor=(0.95, 0.5), 
            title="Domains", fontsize=16, title_fontsize=18)
        
    # plt.tight_layout()

    plt.savefig(f"{base_path}/tasks_compose.pdf", format='pdf')
    plt.close()

def main():
    # show_domains(domains_compose, labels)
    show_tasks(domains_compose, labels)

if __name__ == '__main__':
    main()
    