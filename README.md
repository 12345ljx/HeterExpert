

# HEMoE:Building Mixture-of-Experts with Heterogeneous Emergent Modularity. 

This repository is the official implementation of HEMoE:Building Mixture-of-Experts with Heterogeneous Emergent Modularity. 


## Requirements

To install requirements:

```shell
conda env create -f environment.yml
conda activate HeterExpert
pip install -r requirements.txt
```



## Preparation
Clone code:
```shell
git clone https://github.com/12345ljx/HeterExpert.git
cd ./HeterExpert
```
Download the required model and data from the [huggingface](https://huggingface.co/) website:
```shell
python ./datasets/data_download.py
python ./models/model_download.py
```

Processing raw data to get instruction data:
```shell
python ./Cluster/instance_prompt.py
```
Cluster the data:
```shell
python ./Cluster/gen_domains_cluster.py
```
Collecting importance scores:

```shell
./Neuron_Score/get_score.sh
```
Preprocessing score:

```shell
python ./Neuron_Score/process_score.py
```



## Module partitioning

HEMoE partitioning:
```shell
python ./Split/ilp_split/pre_cluster.py
python ./Split/ilp_split/ilp_solver.py
python ./Split/ilp_split/process_raw_data.py
```

MoEfication partitioning:
```shell
./Split/cluster_split.sh
```

MoEBERT partitioning:
```shell
python ./Split/moebert_split/split.py
```

Random partitioning:
```shell
python ./Split/random_split/random_split_homo.py
python ./Split/random_split/random_split_hetero.py
```



## Training

First, install [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory?tab=readme-ov-file#installation):

```shell
cd LLaMA-Factory
pip install -e ".[torch,metrics]" --no-build-isolation
cd ..
```

To train the model(s) in the paper, run this command:

```train
./LLaMA-Factory/scripts/train_top_k_func.sh
```



## Evaluation

First, install [lm_eval](https://github.com/EleutherAI/lm-evaluation-harness?tab=readme-ov-file#install):

```shell
cd lm-evaluation-harness
pip install -e .
cd ..
```

To evaluate our model, run:

```eval
python ./Eval/eval_main_moe.py
```



## Results

Our model achieves the following performance :

| Method        | ARC-e | ARC-c | PIQA  | OBQA  | WG    | SciQ  | SIQA  | Average |
| ------------- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ------- |
| LLaMA-3.2-1B  | 60.65 | 36.35 | 74.48 | 37.20 | 60.70 | 88.40 | 42.94 | 57.24   |
| Random(hom.)  | 48.86 | 30.63 | 64.09 | 37.00 | 54.62 | 92.20 | 46.01 | 53.34   |
| Random(het.)  | 52.74 | 30.12 | 65.13 | 37.80 | 56.67 | 92.20 | 45.60 | 54.32   |
| MoEBERT       | 45.16 | 27.90 | 62.19 | 34.40 | 52.09 | 90.70 | 42.32 | 50.68   |
| MoEBERT(mul.) | 45.79 | 27.39 | 61.70 | 33.60 | 52.96 | 90.40 | 41.51 | 50.48   |
| MoEfication   | 51.09 | 30.89 | 65.45 | 37.40 | 55.80 | 92.10 | 45.80 | 54.08   |
| EMoE          | 46.72 | 30.21 | 63.06 | 37.00 | 56.83 | 86.10 | 43.30 | 51.89   |
| HEMoE         | 53.66 | 32.85 | 65.83 | 37.80 | 56.67 | 93.00 | 47.08 | 55.27   |

| Method        | ARC-e | ARC-c | PIQA  | OBQA  | WG    | SciQ  | SIQA  | Average |
| :------------ | :---- | :---- | ----- | ----- | ----- | ----- | ----- | ------- |
| LLaMA-3.2-3b  | 71.59 | 46.08 | 77.48 | 43.00 | 69.85 | 92.70 | 46.98 | 63.95   |
| Random(hom.)  | 58.63 | 37.54 | 68.99 | 42.60 | 65.83 | 94.00 | 51.74 | 59.90   |
| Random(het.)  | 59.51 | 38.40 | 69.64 | 42.60 | 66.61 | 94.10 | 51.28 | 60.31   |
| MoEBERT       | 46.17 | 28.67 | 64.31 | 37.60 | 56.43 | 91.80 | 44.99 | 52.85   |
| MoEBERT(mul.) | 46.21 | 27.30 | 63.77 | 34.80 | 54.14 | 92.20 | 44.99 | 51.92   |
| MoEfication   | 60.14 | 36.43 | 70.24 | 43.80 | 65.04 | 93.50 | 50.46 | 59.94   |
| EMoE          | 52.36 | 34.56 | 71.33 | 43.60 | 66.93 | 89.70 | 49.95 | 58.35   |
| HEMoE         | 62.37 | 39.08 | 71.87 | 44.80 | 66.77 | 94.70 | 51.84 | 61.63   |



## License

This repository is licensed under the [MIT License](https://github.com/12345ljx/HeterExpert/blob/main/LICENSE).