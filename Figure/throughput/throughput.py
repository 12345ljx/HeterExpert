import numpy as np
import matplotlib.pyplot as plt

plt.switch_backend('agg')
plt.rcParams['pdf.use14corefonts'] = True
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica']  # 'Arial' 'Helvetica'

def batch_size():
    # seq_len = 512
    batchsizes = [32, 64, 128, 256]
    data_2_8 = [880.01, 933.59, 962.17, 975.44]
    data_4_8 = [805.18, 853.32, 877.58, 887.60]
    data_6_8 = [734.72, 774.92, 794.27, 800.58]
    data_mlp = [680.68, 709.18, 721.57, 728.25]
    return batchsizes, data_2_8, data_4_8, data_6_8, data_mlp
    
def seq_len():
    # batch_size = 64
    seq_lens = [256, 512, 768, 1024]
    data_2_8 = [1624.69, 933.59, 643.50, 485.39]
    data_4_8 = [1515.02, 853.32, 586.01, 442.60]
    data_6_8 = [1390.40, 774.92, 530.83, 401.05]
    data_mlp = [1355.51, 709.18, 475.25, 355.17]
    return seq_lens, data_2_8, data_4_8, data_6_8, data_mlp
    
def plot(x, y1, y2, y3, y4, xlabel, ylabel):
    bar_width = 0.2
    index = np.arange(len(x))
    fig, ax = plt.subplots(figsize=(15, 6))
    fig.subplots_adjust(left=0.15, bottom=0.2, top=0.89, right=0.95)
    
    ax.bar(index - bar_width * 1, y1, bar_width, edgecolor='black', zorder=2, color="#8b0000", label='Ratio=40%')
    ax.bar(index + bar_width * 0, y2, bar_width, edgecolor='black', zorder=2, color="#cc0000", label='Ratio=55%')
    ax.bar(index + bar_width * 1, y3, bar_width, edgecolor='black', zorder=2, color="#ff6666", label='Ratio=70%')
    ax.bar(index + bar_width * 2, y4, bar_width, edgecolor='black', zorder=2, color="#a9a9a9", label='Original')
    ax.grid(True, linestyle='--', color='gray', alpha=0.5, zorder=1, axis='y')
    
    ax.tick_params(axis='x', labelsize=35)
    ax.tick_params(axis='y', labelsize=35)
    label_font = {
        'fontsize': 35,
        'fontweight': 'bold',
    }
    labelpad = 8
    ax.set_xlabel(xlabel, fontdict=label_font, labelpad=labelpad, loc='center')
    ax.set_ylabel(ylabel, fontdict=label_font, labelpad=labelpad, loc='center')

    if xlabel == 'Batch Size':
        ax.set_ylim(ymin=600)
        ax.set_yticks(np.arange(600, 1001, 100))
        ax.set_yticklabels(['600', '700', '800', '900', '1000'])
    elif xlabel == 'Sequence Length':
        ax.set_ylim(ymin=300)
        ax.set_yticks(np.arange(500, 1501, 250))
        ax.set_yticklabels(['500', '750', '1000', '1250', '1500'])

    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(x)
    ax.legend(
        fontsize=18,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.15), 
        ncol=4, 
        frameon=True
    )

    plt.savefig(f"/usr/workdir/HeterExpert/Figure/throughput/{xlabel}.pdf", format='pdf')
    plt.close()
    
    
def main():
    plot(*batch_size(), xlabel='Batch Size', ylabel='Tokens / sec')
    plot(*seq_len(), xlabel='Sequence Length', ylabel='Tokens / sec')

if __name__ == '__main__':
    main()