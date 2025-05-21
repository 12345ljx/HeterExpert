import torch
import torch.nn as nn
import torch.nn.functional as F

class TopKGate(nn.Module):
    """A custom layer for selecting a sparse mixture of experts.

  Let f_1, f_2, ..., f_n be the experts. The layer returns:

              a_1 * f_1 + a_2 * f_2 + ... + a_n * f_n,

  where the mixture weights satisfy a_1, ..., a_n is 1 or 0 and a_1 + ... + a_n = k.
  The number of non-zeros in the mixture weights can be directly controlled.
  The layer is differentiable and can be trained using first-order methods like
  SGD.

  Input: For task-only conditioning, the input should be a list of tensors,
    each corresponding to an expert. 
    Note: In both cases, the expert tensors should have the same shape.
  Output: Tensor, with the same shape as the expert tensors.

  Example:
    # Construct a DSelectKGate to select 2 out of 4 experts.
    gate = TopKGate(num_selects=2)
    # output_tensor is a sparse mixture of the 4 tensors in the inputs.
    output_tensor = gate(inputs)
  """
    def __init__(self, hidden_size, num_selects, num_experts, static=True, balance_loss_weight=None, size_rate_experts=None, add_noise=False, w_initializer=None):
        super(TopKGate, self).__init__()
        self.num_selects = num_selects
        self.num_experts = num_experts
        self.static = static
        self.w_initializer = w_initializer or (lambda x : nn.init.normal_(x, mean=0, std=0.001))
        self.reg_loss = 0
        
        self.use_balance = True if balance_loss_weight else False
        self.balance_loss_weight = balance_loss_weight
        
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.size_rate_experts = torch.tensor(size_rate_experts, device=device)
        
        self.add_noise = add_noise
        self.noise_epsilon = 1e-6
        
        if self.static:
            self.w_logits = nn.Parameter(self.w_initializer(torch.empty(self.num_experts)))
        else:
            self.w_logits = nn.Linear(hidden_size, self.num_experts, bias=False)
            self.w_initializer(self.w_logits.weight)
            if self.add_noise:
                self.weight_noise = nn.Linear(hidden_size, num_experts, bias=False)
                self.weight_noise.weight.data = torch.zeros(
                    (num_experts, hidden_size),
                    requires_grad=True,
                    device=self.weight_noise.weight.data.device,
                    dtype=self.weight_noise.weight.data.dtype,
                )
    
        
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
        if self.use_balance or self.add_noise:
            raise NotImplementedError('add_noise and use_balance are not implemented yet for task-level routing')
            
        context = torch.enable_grad() if self.training else torch.no_grad()
        with context:
            top_k_logits, top_k_indices = torch.topk(input=self.w_logits, k=self.num_selects)
            top_k_scores = 1 + top_k_logits - top_k_logits.detach()
            expert_weights = torch.zeros_like(self.w_logits)
            expert_weights = expert_weights.scatter(dim=0, index=top_k_indices, src=top_k_scores)
            
        return expert_weights

    def _compute_example_conditioned_expert_weights(self, routing_inputs):
        # routing_inputs.shape [num_tokens, hidden_size]
        context = torch.enable_grad() if self.training else torch.no_grad()
        with context:
            logits_gate = self.w_logits(routing_inputs)  # [num_tokens, num_experts]
            if self.training and self.add_noise:
                noise_mm = self.weight_noise(routing_inputs)  # [num_tokens, num_experts]
                noise_control = F.softplus(noise_mm) + self.noise_epsilon
                logits_noise = torch.randn_like(logits_gate) * noise_control
                logits = logits_gate + logits_noise
            else:
                logits = logits_gate
            
            if self.training:
                logits = F.softmax(logits.to(torch.float32), dim=1).to(logits.dtype)
                top_k_logits, top_k_indices = logits.topk(min(self.num_selects, self.num_experts), dim=1)  # [num_tokens, num_selects]
                top_k_scores = 1 + top_k_logits - top_k_logits.detach()
                zeros = torch.zeros_like(logits, requires_grad=True, device=logits.device)
                expert_weights = zeros.scatter(dim=1, index=top_k_indices, src=top_k_scores)  # [num_tokens, num_experts]
            else:
                logits = F.softmax(logits.to(torch.float32), dim=1).to(logits.dtype)
                _, top_k_indices = logits.topk(self.num_selects, dim=1)  # [num_tokens, num_selects]
                expert_weights = torch.zeros_like(logits, requires_grad=False, device=logits.device)
                expert_weights.scatter_(dim=1, index=top_k_indices, value=1.0)  # [num_tokens, num_experts]
            
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
