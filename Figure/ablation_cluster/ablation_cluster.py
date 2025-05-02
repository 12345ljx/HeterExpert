import numpy as np
import matplotlib.pyplot as plt

plt.switch_backend('agg')
plt.rcParams['pdf.use14corefonts'] = True
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica']  # 'Arial' 'Helvetica'

in_domain = ['ARC-e', 'ARC-c', 'PIQA', 'OBQA', 'WG', 'SciQ', 'SIQA', 'Average']

clutering = [47.31, 27.65, 64.31, 37.00, 58.33, 87.10, 46.26, 52.57]
task_type = [48.15, 28.41, 64.96, 38.00, 59.04, 86.70, 46.21, 53.07]

bar_width = 0.4
index = np.arange(len(in_domain))
fig, ax = plt.subplots(figsize=(15, 8))
fig.subplots_adjust(left=0.1,bottom=0.15,top=0.95,right=0.95)

ax.bar(index, clutering, bar_width, color="#fdd39f", label='Clustering')
ax.bar(index + bar_width, task_type, bar_width, color="#ee7a5f", label='Task Category')

ax.tick_params(axis='x', labelsize=18)
ax.tick_params(axis='y', labelsize=18)
label_font = {
    'fontsize': 23,
    'fontweight': 'bold',
}
labelpad = 8
ax.set_xlabel('Tasks', fontdict=label_font, labelpad=labelpad, loc='center')
ax.set_ylabel('Performance', fontdict=label_font, labelpad=labelpad, loc='center')

ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(in_domain)
ax.legend(fontsize=18)

plt.savefig("/usr/workdir/HeterExpert/Figure/ablation_cluster/ablation_cluster.pdf", format='pdf')
plt.close()