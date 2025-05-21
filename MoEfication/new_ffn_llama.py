import types
import torch
import torch.nn as nn
from transformers.models.llama.modeling_llama import LlamaMLP

from MoEfication.gate_top_k import TopKGate
from MoEfication.gate_top_p import TopPGate
from MoEfication.gate_dynk_max import DynkMaxGate
from MoEfication.moe_triton_impls import MoeFirstLayerGLUHeteroImplementation, MoeSecondLayerMergingHeteroImplementation

def get_act_name(act_fn):
    if isinstance(act_fn, nn.SiLU):
        return 'silu'
    else:
        raise ValueError

def moe_ffn_llama(ffn, labels, moeargs):
    assert type(ffn) == LlamaMLP
    size_experts = [len(neurons) for neurons in labels]
    ffn.is_moe = True
    ffn.num_selected_expert = moeargs.num_selected_expert
    ffn.num_expert = moeargs.num_expert
    ffn.static = moeargs.static
    ffn.activation_str = get_act_name(ffn.act_fn)
    ffn.forward_old = ffn.forward
    ffn.experts_in = nn.ModuleList()
    ffn.experts_gate = nn.ModuleList()
    ffn.experts_out = nn.ModuleList()
    
    wi_tensor = ffn.up_proj.weight.data  # [dff_hidden_size, hidden_size]
    gate_tensor = ffn.gate_proj.weight.data  # [dff_hidden_size, hidden_size]
    wo_tensor = ffn.down_proj.weight.data  # [hidden_size, dff_hidden_size]
    # print(wi_tensor.size())
    
    hidden_size = wo_tensor.size(0)
    dff_hidden_size = wo_tensor.size(1)
    size_rate_experts = [i / dff_hidden_size for i in size_experts]
    
    for i in range(ffn.num_expert):
        wi_expert = wi_tensor[labels[i], :]
        gate_expert = gate_tensor[labels[i], :]
        wo_expert = wo_tensor[:, labels[i]]
        # print(wi_expert)
        # print(wo_expert)
        
        wi_layer = torch.nn.Linear(hidden_size, size_experts[i], bias=False)
        wi_layer.weight.data = wi_expert
        ffn.experts_in.append(wi_layer)
        
        gate_layer = torch.nn.Linear(hidden_size, size_experts[i], bias=False)
        gate_layer.weight.data = gate_expert
        ffn.experts_gate.append(gate_layer)
        
        wo_layer = torch.nn.Linear(size_experts[i], hidden_size, bias=False)
        wo_layer.weight.data = wo_expert
        ffn.experts_out.append(wo_layer)
        
    if moeargs.gate_mode == 'top_k':
        ffn.gate = TopKGate(hidden_size=hidden_size, num_selects=moeargs.num_selected_expert, num_experts=moeargs.num_expert, static=moeargs.static,
                            balance_loss_weight=moeargs.balance_loss_weight, size_rate_experts=size_rate_experts)
    elif moeargs.gate_mode == 'top_p':
        ffn.gate = TopPGate(hidden_size=hidden_size, top_p_threshold=moeargs.top_p_threshold, num_experts=moeargs.num_expert, static=moeargs.static, 
                            balance_loss_weight=moeargs.balance_loss_weight, size_rate_experts=size_rate_experts)
    elif moeargs.gate_mode == 'dynk_max':
        ffn.gate = DynkMaxGate(hidden_size=hidden_size, tau=moeargs.tau, num_experts=moeargs.num_expert, static=moeargs.static, 
                            balance_loss_weight=moeargs.balance_loss_weight, size_rate_experts=size_rate_experts)
    else:
        raise NotImplementedError
    
    def expert(ffn_self, i, hidden_states):
        # print('computing expert', i, 'input shape: ', hidden_states.shape, 'expert size:', size_experts[i])
        gate_dff = ffn_self.act_fn(ffn_self.experts_gate[i](hidden_states))  # [..., size_experts[i]]
        hidden_states_dff = ffn_self.experts_in[i](hidden_states)
        expert_output = ffn_self.experts_out[i](gate_dff * hidden_states_dff)
        assert expert_output.shape[-1] == hidden_size
        return expert_output 
    
    ffn.expert = types.MethodType(expert, ffn)
    
    del ffn.gate_proj
    del ffn.up_proj
    del ffn.down_proj
    
    def new_forward(ffn_self, hidden_states):
        assert moeargs.gate_mode == 'top_k', 'only top_k is supported for now'
        expert_outputs = [ffn_self.expert(i, hidden_states) for i in range(ffn_self.num_expert)]
            
        if ffn_self.static:    
            _, expert_weights = ffn_self.gate()
            hidden_states = torch.sum(torch.stack([expert_weights[i] * expert_outputs[i] for i in range(ffn_self.num_expert)], dim=0), dim=0)
        else:
            ori_shape = hidden_states.shape
            expert_outputs = [expert_outputs[i].view(-1, hidden_size) for i in range(ffn_self.num_expert)]
            hidden_states = hidden_states.view(-1, hidden_size)
            
            _, expert_weights = ffn_self.gate(hidden_states)
            experts_output = torch.stack([expert_weights[:, i].view(-1, 1) * expert_outputs[i] for i in range(ffn_self.num_expert)], dim=0)  # [num_experts, num_tokens, hidden_size]
            hidden_states = torch.sum(experts_output, dim=0)  # [num_tokens, hidden_size]
            hidden_states = hidden_states.view(ori_shape)
            
        return hidden_states
    
    def new_forward_index(ffn_self, hidden_states):
        assert not ffn_self.training, 'only support inference for now'
        assert moeargs.gate_mode == 'top_k', 'only top_k is supported for now'
        if ffn_self.static: 
            _, expert_weights = ffn_self.gate()
            chosen_experts_idx = torch.where(expert_weights > 0)[0]
            experts_output = torch.stack([expert_weights[i] * ffn_self.expert(i, hidden_states) for i in chosen_experts_idx], dim=0)  # [num_selects, batchsize, seq_len, hidden_size]
            hidden_states = torch.sum(experts_output, dim=0)  # [batchsize, seq_len, hidden_size]
        else:
            ori_shape = hidden_states.shape  # [batchsize, seq_len, hidden_size]
            hidden_states = hidden_states.view(-1, hidden_size)
            experts_output = torch.zeros_like(hidden_states)  # [num_tokens, hidden_size]
            
            _, expert_weights = ffn_self.gate(hidden_states)  # [num_tokens, num_experts]
            for i in range(ffn_self.num_expert):
                input_idx = torch.where(expert_weights[:, i] > 0)[0]
                if len(input_idx) == 0: continue
                experts_output[input_idx] += ffn_self.expert(i, hidden_states[input_idx]) * expert_weights[input_idx, i, None]
            hidden_states = experts_output.view(ori_shape)
            
        return hidden_states
    
    if not ffn.training and moeargs.gate_backend is not None:
        max_expert_size = max(size_experts)
        ffn.size_experts = nn.Parameter(torch.tensor(size_experts, dtype=torch.float32))
        ffn.gate_proj_pad = nn.Parameter(torch.zeros(ffn.num_expert, hidden_size, max_expert_size))
        ffn.up_proj_pad = nn.Parameter(torch.zeros(ffn.num_expert, hidden_size, max_expert_size))
        ffn.down_proj_pad = nn.Parameter(torch.zeros(ffn.num_expert, max_expert_size, hidden_size))
        with torch.no_grad():
            for i, expert_size in enumerate(size_experts):
                ffn.gate_proj_pad[i, :, :expert_size] = ffn.experts_gate[i].weight.t()
                ffn.up_proj_pad[i, :, :expert_size] = ffn.experts_in[i].weight.t()
                ffn.down_proj_pad[i, :expert_size, :] = ffn.experts_out[i].weight.t()
    
    @torch.jit.script
    def extract_indices(routing_tensor):
        routing_tensor = routing_tensor.to(torch.int32)
        sort_indices = routing_tensor.argsort(dim=0, descending=True)
        expert_bincounts = routing_tensor.sum(dim=0)
        unsort_indices = sort_indices.argsort(dim=0)
        return sort_indices, unsort_indices, expert_bincounts
    
    def new_forward_triton(ffn_self, hidden_states):
        assert not ffn_self.training, 'only support inference for now'
        assert moeargs.gate_mode == 'top_k', 'only top_k is supported for now'
        assert not ffn_self.static, 'only dynamic is supported for now'
        
        ori_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_size)
        _, expert_weights = ffn_self.gate(hidden_states)  # [num_tokens, num_experts]
        with torch.no_grad():
            sort_indices, unsort_indices, expert_bincounts = extract_indices(expert_weights)
        intermediate_acts = MoeFirstLayerGLUHeteroImplementation.apply(hidden_states, 
                                                                       ffn_self.gate_proj_pad, 
                                                                       ffn_self.up_proj_pad, 
                                                                       sort_indices, 
                                                                       expert_bincounts, 
                                                                       ffn_self.size_experts.to(torch.int32), 
                                                                       ffn_self.activation_str)
        experts_output = MoeSecondLayerMergingHeteroImplementation.apply(intermediate_acts, 
                                                                    ffn_self.down_proj_pad, 
                                                                    unsort_indices, 
                                                                    expert_bincounts, 
                                                                    ffn_self.size_experts.to(torch.int32),
                                                                    expert_weights)
        hidden_states = experts_output.view(ori_shape)
        return hidden_states
    
    if moeargs.gate_backend == 'index':
        ffn.forward = types.MethodType(new_forward_index, ffn)
    elif moeargs.gate_backend == 'triton':
        ffn.forward = types.MethodType(new_forward_triton, ffn)
    else:
        ffn.forward = types.MethodType(new_forward, ffn)