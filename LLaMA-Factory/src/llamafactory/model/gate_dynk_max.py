import torch
import torch.nn as nn
import torch.nn.functional as F

EPSILON = 1e-6

class DynkMaxGate(nn.Module):
    def __init__(self, hidden_size, tau, num_experts, static=True, balance_loss_weight=None, size_rate_experts=None):
        super(DynkMaxGate, self).__init__()
        self.tau = tau
        self.num_experts = num_experts
        self.static = static
        self.w_initializer = lambda x : nn.init.normal_(x, mean=0, std=0.001)
        self.reg_loss = 0
        
        self.use_balance = True if balance_loss_weight else False
        self.balance_loss_weight = balance_loss_weight
        
        self.size_rate_experts = nn.Parameter(torch.tensor(size_rate_experts))
        
        if self.static:
            self.w_logits = nn.Parameter(self.w_initializer(torch.empty(self.num_experts)))
        else:
            self.w_logits = nn.Linear(hidden_size, self.num_experts, bias=False)
            self.w_initializer(self.w_logits.weight)
    
        
    def forward(self, routing_inputs=None):
        self.reg_loss = 0
        
        if self.static:  # Task-only routing.
            expert_weights = self._compute_expert_weights()
        else:  # Example-conditioned routing.
            expert_weights = self._compute_example_conditioned_expert_weights(routing_inputs)  # [num_tokens, num_experts]

        if self.training:
            self._add_regularization_loss(expert_weights)

        output = None
        return output, expert_weights

    def _compute_expert_weights(self):
        raise NotImplementedError

    def _compute_example_conditioned_expert_weights(self, routing_inputs):
        # routing_inputs.shape [num_tokens, hidden_size]
        context = torch.enable_grad() if self.training else torch.no_grad()
        with context:
            logits = self.w_logits(routing_inputs)  # [num_tokens, num_experts]
            
            logits = F.softmax(logits.to(torch.float32), dim=1).to(logits.dtype)
            max_norms, _ = logits.max(dim=-1, keepdim=True)  # [num_tokens, 1]
            thresholds = max_norms * self.tau
            
            logits_score = 1 + logits - logits.detach()
            expert_weights = logits_score.masked_fill(logits < thresholds, 0.0)
            
            if self.training and self.use_balance:
                token_frac = expert_weights.mean(dim=0)  # [num_experts]
                token_frac = token_frac * self.size_rate_experts.to(token_frac.dtype)  # parameter penalty balance loss
                prob_frac = logits.mean(dim=0)
                num_tokens = logits.size(0)
                balance_loss = (token_frac * prob_frac).sum() * num_tokens
                self.balance_loss = balance_loss * self.balance_loss_weight
        
        return expert_weights

    def _add_regularization_loss(self, expert_weights):
        with torch.enable_grad():
            if not self.static:  # add load balancing loss
                # self.reg_loss += self.entropy_reger_balancing(expert_weights)
                if self.use_balance:
                    self.reg_loss += self.balance_loss

# Example usage:
# gate = TopKGate(num_selects=2)
# output_tensor = gate((input_tensor_list, routing_inputs))
