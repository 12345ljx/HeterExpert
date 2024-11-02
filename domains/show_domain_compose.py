import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

domains_compose = np.load('/usr/workdir/HeterExpert/domains/domains_compose.npy')

tasks_list = []
directory = Path('/usr/workdir/HeterExpert/data')
subdirs = [d for d in directory.iterdir() if d.is_dir()]
for task_path in subdirs:
    task_name = task_path.name.replace('_template', '')
    tasks_list.append(task_name)
labels = sorted(tasks_list)

domains_num = 8
tasks_num = len(labels)

def show_domains(domains_compose: np.ndarray, labels: list):
    colors = plt.cm.plasma(np.linspace(0, 1, tasks_num))
    fig, axs = plt.subplots(2, 4, figsize=(15, 5))
    for ax, domain in zip(axs.flat, range(domains_num)):
        domain_compose = domains_compose[domain]
        ax.pie(domain_compose, labels=None, colors=colors, autopct=None, startangle=140)
        ax.axis('equal')
        ax.set_title('Domain {}'.format(domain))

    plt.subplots_adjust(right=0.75)

    handles = [plt.Line2D([0], [0], marker='o', color='w', label=label, markersize=10, markerfacecolor=color) 
            for label, color in zip(labels, colors)]
    fig.legend(handles=handles, loc='center right', bbox_to_anchor=(0.95, 0.5), 
            title="Tasks", ncol=2)
        
    # plt.tight_layout()

    plt.savefig(f"/usr/workdir/HeterExpert/domains/domains_compose.pdf", format='pdf')
    plt.close()
    
def show_tasks(domains_compose: np.ndarray, labels: list):
    tasks_compose = np.transpose(domains_compose)
    
    colors = plt.cm.plasma(np.linspace(0, 1, domains_num))
    fig, axs = plt.subplots(4, 7, figsize=(20, 12))
    for ax, task in zip(axs.flat, range(tasks_num)):
        task_compose = tasks_compose[task]
        ax.pie(task_compose, labels=None, colors=colors, autopct=None, startangle=140)
        ax.axis('equal')
        ax.set_title('Task {}'.format(labels[task]))
    
    for j in range(26, 28):
        fig.delaxes(axs.flatten()[j])

    plt.subplots_adjust(right=0.8)

    domains_label = [f"Domain {i}" for i in range(domains_num)]
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=label, markersize=10, markerfacecolor=color) 
            for label, color in zip(domains_label, colors)]
    fig.legend(handles=handles, loc='center right', bbox_to_anchor=(0.95, 0.5), 
            title="Domains", ncol=2)
        
    # plt.tight_layout()

    plt.savefig(f"/usr/workdir/HeterExpert/domains/tasks_compose.pdf", format='pdf')
    plt.close()

def main():
    show_domains(domains_compose, labels)
    # show_tasks(domains_compose, labels)

if __name__ == '__main__':
    main()
    