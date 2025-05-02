import numpy as np

def is_binary(tensor):
    return np.all(np.logical_or(np.isclose(tensor, 0, atol=1e-6), np.isclose(tensor, 1, atol=1e-6)))

class ExpertObserve :
    def __init__(self, relative_expert_size) :
        self.relative_expert_size = relative_expert_size  # [num_layers, num_experts]
    
    def observe_llama(self, model, result, attention_mask) :
        def forward_observe(layer) :
            def fn(intermediate, input, output) :
                nonlocal attention_mask
                attention_mask = attention_mask.view(-1)
                expert_weights = output[1].clone()  # [num_tokens, num_experts]
                expert_weights[attention_mask == 0, :] = 0.0
                expert_weights = expert_weights.cpu().numpy()
                
                expert_weights = np.round(expert_weights).astype(int)
                params_rate = expert_weights * self.relative_expert_size[layer]
                result[layer] += np.sum(params_rate, axis=1)
            return fn
        
        hooks = []
        for layer in range(model.config.num_hidden_layers):
            if not hasattr(model.model.layers[layer].mlp, 'gate'):
                continue
            intermediate = model.model.layers[layer].mlp.gate
            hook = intermediate.register_forward_hook(forward_observe(layer))
            hooks.append(hook)
        return hooks
    
    def observe_library(self, model_name) :
        if model_name in ['llama3.2-1b', 'llama3.2-3b']:
            return self.observe_llama
        else:
            print(f'Cant find an observation function that matched the model {model_name}')
            raise NotImplementedError
        
    def erase_hooks(self, hooks) :
        for hook in hooks :
            hook.remove()