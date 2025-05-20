

# HEMoE:Building Mixture-of-Experts with Heterogeneous Emergent Modularity. 

This repository is the official implementation of HEMoE:Building Mixture-of-Experts with Heterogeneous Emergent Modularity. 


## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

>?  Describe how to set up the environment, e.g. pip/conda/docker commands, download datasets, etc...

## 得到专家

### 准备数据
从huggingface下载原始数据，并处理原始数据得到指令数据
```python
python ./Cluster/instance_prompt.py
```
对数据进行聚类
```python
python ./Cluster/gen_domains_cluster.py
```
### 收集重要性得分
```shell
./Neuron_Score/get_score.sh
```
## 专家划分
神经元预聚类
```python
python ./Split/ilp_split/pre_cluster.py
```
专家划分
```python
python ./Split/ilp_split/ilp_solver.py
python ./Split/ilp_split/process_raw_data.py
```


## Training

To train the model(s) in the paper, run this command:

```train
./LLaMA-Factory/scripts/train_top_k_func.sh
```

## Evaluation

To evaluate my model on ImageNet, run:

```eval
python eval.py --model-file mymodel.pth --benchmark imagenet
```

>?  Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).

## Pre-trained Models

You can download pretrained models here:

- [My awesome model](https://drive.google.com/mymodel.pth) trained on ImageNet using parameters x,y,z. 

>?  Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models.

## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

>?  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## Contributing

>?  Pick a licence and describe how to contribute to your code repository. 