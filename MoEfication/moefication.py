
import os
from typing import Literal, Optional, Any
import torch
import numpy as np
from dataclasses import dataclass, asdict
from transformers.utils import is_torch_cuda_available

from MoEfication.new_ffn_llama import moe_ffn_llama


def get_current_device() -> torch.device:
    if is_torch_cuda_available():
        device = "cuda:{}".format(os.environ.get("LOCAL_RANK", "0"))
    else:
        device = "cpu"
    return torch.device(device)

@dataclass
class MoEArgs:
    model_name: Optional[str] = None
    function_name: Optional[str] = None
    split_mode: Literal['random', 'random_hetero', 'co_act', 'co_act(average)', 'ilp', 'ilp(non_log)', 'cluster', 'moebert', 'moebert_multitask'] = 'co_act'
    gate_mode: Optional[Literal['dselect_k', 'random', 'top_k', 'top_p', 'dynk_max', 'ground_truth', 'mlp']] = None
    gate_backend: Optional[Literal['mask', 'index', 'triton']] = None
    static: bool = True
    begin_layer: int = 4
    end_layer: int = -1
    num_expert: Literal[8, 16, 32, 64, 128, 256] = 32
    num_selected_expert: Optional[int] = None
    gamma: Optional[float] = None
    threshold: Optional[int] = None
    top_p_threshold: Optional[float] = None
    tau: Optional[float] = None
    balance_loss_weight: Optional[float] = None
    
    def __post_init__(self):
        assert self.split_mode in ['random', 'random_hetero', 'co_act', 'co_act(average)', 'ilp', 'ilp(non_log)', 'cluster', 'moebert', 'moebert_multitask']
        if self.gate_mode is not None:
            assert self.gate_mode in ['dselect_k', 'random', 'top_k', 'top_p', 'dynk_max', 'ground_truth', 'mlp']
        
        assert self.num_expert > 0
        if self.num_selected_expert:
            assert self.num_selected_expert > 0
            assert self.num_expert >= self.num_selected_expert
        
        if self.threshold:
            assert self.threshold > 0
            
        if self.gate_mode == 'dselect_k':
            assert self.gamma is not None
        elif self.gate_mode == 'top_p':
            assert self.top_p_threshold is not None
            assert 0 <= self.top_p_threshold and self.top_p_threshold <= 1
        elif self.gate_mode == 'dynk_max':
            assert self.tau is not None
            assert 0 <= self.tau and self.tau <= 1
            
        self.layer_range = range(self.begin_layer - 1, self.end_layer)
        
    def get_split_str(self) -> str:
        if self.split_mode in ['cluster', 'random', 'random_hetero']:
            res = f'{self.split_mode}'
        else:
            res = f'{self.split_mode}[{self.function_name}]'
        
        return res
    
    def __str__(self) -> str:
        if self.balance_loss_weight is not None:
            balance_loss_weight_str = f'{self.balance_loss_weight:f}'.rstrip('0')
        else:
            balance_loss_weight_str = '0'
        if self.gate_mode in ['top_k', 'mlp']:
            res = f'split={self.get_split_str()},gate={self.gate_mode}/n{self.num_expert}k{self.num_selected_expert}\
(static={self.static},layer={self.begin_layer}-{self.end_layer},balance_loss_weight={balance_loss_weight_str})'
        elif self.gate_mode == 'dselect_k':
            res = f'split={self.get_split_str()},gate={self.gate_mode}/n{self.num_expert}k{self.num_selected_expert}\
(static={self.static},layer={self.begin_layer}-{self.end_layer},gamma={self.gamma})'
        elif self.gate_mode == 'top_p':
            res = f'split={self.get_split_str()},gate={self.gate_mode}/n{self.num_expert}p{self.top_p_threshold}\
(static={self.static},layer={self.begin_layer}-{self.end_layer},balance_loss_weight={balance_loss_weight_str})'
        elif self.gate_mode == 'dynk_max':
            res = f'split={self.get_split_str()},gate={self.gate_mode}/n{self.num_expert}t{self.tau}\
(static={self.static},layer={self.begin_layer}-{self.end_layer},balance_loss_weight={balance_loss_weight_str})'
        else:
            raise NotImplementedError
        return res
    
    def get_labels_base_path(self) -> str:
        model_name = self.model_name
        function_name = self.function_name
        split_mode = self.split_mode
        
        if 'co_act' in split_mode:
            base_path = '/usr/workdir/MoEfication/moefication/get_hidden/model_split/co_activations'
            if 'average' in split_mode:
                base_path = os.path.join(base_path, 'averaged')
            base_path = os.path.join(base_path, f'{model_name}/{function_name}')
        elif split_mode == 'ilp':
            base_path = '/usr/workdir/HeterExpert/Split/model_split/ilp'
            base_path = os.path.join(base_path, f'{model_name}/{function_name}')
        elif split_mode == 'ilp(non_log)':
            base_path = '/usr/workdir/HeterExpert/Split/model_split/ilp(non_log)'
            base_path = os.path.join(base_path, f'{model_name}/{function_name}')
        elif split_mode == 'random':
            base_path = f'/usr/workdir/MoEfication/moefication/get_hidden/model_split/random/{model_name}'
        elif split_mode == 'random_hetero':
            base_path = f'/usr/workdir/HeterExpert/Split/model_split/random_hetero/{model_name}'
        elif split_mode == 'moebert':
            base_path = '/usr/workdir/HeterExpert/Split/model_split/moebert'
            base_path = os.path.join(base_path, f'{model_name}/{function_name}')
        elif split_mode == 'moebert_multitask':
            base_path = '/usr/workdir/HeterExpert/Split/model_split/moebert_multitask'
            base_path = os.path.join(base_path, f'{model_name}/{function_name}')
        elif split_mode == 'cluster':
            base_path = f'/usr/workdir/HeterExpert/Split/model_split/cluster/{model_name}'
        
        print("load labels from:", base_path)
        self.base_path = base_path
        return base_path
    
    @staticmethod
    def read_labels(labels_path) -> list[np.ndarray]:
        num_expert = int(labels_path.split('.')[-1])
        labels = [[] for _ in range(num_expert)]
        if not os.path.exists(labels_path):
            raise FileExistsError(f"can't find the experts splited file: {labels_path}")
        with open(labels_path) as fin:
            for node_idx, expert_idx in enumerate(fin):
                expert_idx = expert_idx.strip()
                if expert_idx == '':
                    continue
                if expert_idx.isdigit():
                    labels[int(expert_idx)].append(node_idx)
                else:
                    for e in expert_idx.split(','):
                        labels[int(e)].append(node_idx)
            
        labels_array = [np.array(labels[i]) for i in range(num_expert)]
        return labels_array
    
    def get_labels_dict(self) -> dict[int, np.ndarray]:
        labels = {}
        base_path = self.get_labels_base_path()
        print("loading labels from:", base_path)
        for layer in self.layer_range:
            labels_layer_path = os.path.join(base_path, f'{self.num_expert}/{layer}.part.{self.num_expert}')
            labels[layer] = MoEArgs.read_labels(labels_layer_path)
            
        return labels
    
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

def moeficate(model, moeargs: MoEArgs):
    if moeargs.num_selected_expert is not None:
        print(f'moeficating the model: n={moeargs.num_expert},k={moeargs.num_selected_expert}')
    elif moeargs.top_p_threshold is not None:
        print(f'moeficating the model: n={moeargs.num_expert},p={moeargs.top_p_threshold}')
    elif moeargs.tau is not None:
        print(f'moeficating the model: n={moeargs.num_expert},tau={moeargs.tau}')
        
    model_name = moeargs.model_name
    base_path = moeargs.get_labels_base_path()
    for layer in moeargs.layer_range:
        if model_name in ["relu-llama2-7b", 'llama3.2-1b', 'llama3.2-3b', 'llama3.2-1b-instruct']:
            ffn_layer = model.model.layers[layer].mlp
            moe_ffn_llama(ffn_layer, MoEArgs.read_labels(os.path.join(base_path, f'{moeargs.num_expert}/{layer}.part.{moeargs.num_expert}')), moeargs)
        else:
            raise NotImplementedError
        
        if isinstance(ffn_layer, list):
            for layer in ffn_layer:
                layer.gate.to(get_current_device()).to(model.dtype)
        else:
            ffn_layer.gate.to(get_current_device()).to(model.dtype)
            
    return model