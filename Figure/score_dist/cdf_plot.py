import seaborn as sns
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import normalize, MinMaxScaler
import matplotlib.pyplot as plt


def read_score(task_list, layer_idx):
    df = pd.DataFrame({})
    scaler = MinMaxScaler(feature_range=(0, 1))
    for task_name in task_list:
        importance_file = f'/usr/workdir/HeterExpert/Neuron_Importance/score/{task_name}/importance_score.pkl'
        with open(importance_file, "rb") as file:
            data = np.array(pickle.load(file))
            data = data[layer_idx]
            data = np.log1p(data)
            # data = scaler.fit_transform(data.reshape(-1, 1)).reshape(-1)
            # data = normalize(data.reshape(-1, 1), axis=0).reshape(-1)
            print(task_name, f'max_value: {np.max(data)}', f'min_value: {np.min(data):.2f}', data.shape)
            df[task_name] = data
    return df

def plot(data):
    plt.figure(figsize=(8, 5))
    sns.kdeplot(data, fill=True, cumulative=False, bw_adjust=1, color="skyblue")
    plt.title("Probability Density Function (KDE)")
    plt.xlabel("Data values")
    plt.ylabel("Density")
    plt.xlim(0, 10)
    plt.grid(alpha=0.3)
    plt.savefig(f'/usr/workdir/HeterExpert/Figure/score_dist/pdf.pdf', format='pdf')
    plt.close()
    
def main():
    task_list = ['arc_easy', 'arc_challenge', 'hellaswag', 'piqa', 'openbookqa', 'winogrande', 'sciq', 'siqa']
    task_list = ['arc_easy', 'arc_challenge', 'hellaswag', 'piqa']
    layer_idx=0
    data = read_score(task_list, layer_idx)
    plot(data)
    
if __name__ == '__main__':
    main()
    