import torch
import torch.nn as nn
import math

class LoRAExpert(nn.Module):
    """单个 LoRA 专家模块"""
    def __init__(self, in_features, out_features, rank, alpha, dropout):
        super().__init__()
        self.lora_down = nn.Linear(in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, out_features, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.scale = alpha / rank
        
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x):
        down = self.lora_down(self.dropout(x))
        up = self.lora_up(down)
        return up * self.scale

class FeRALinear(nn.Module):
    def __init__(self, base_layer: nn.Linear, config):
        super().__init__()
        self.base_layer = base_layer
        self.base_layer.requires_grad_(False)
        
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        
        self.experts = nn.ModuleList([
            LoRAExpert(
                self.in_features, 
                self.out_features, 
                config.rank, 
                config.alpha, 
                config.dropout
            )
            for _ in range(config.num_experts)
        ])
        
        self.routing_weights = None

    def set_routing_weights(self, weights):
        self.routing_weights = weights

    def forward(self, x):
            base_out = self.base_layer(x)
            
            if self.routing_weights is None:
                return base_out
                
            # Experts Output
            # x shape: (B, Seq, Dim) -> expert output: (B, Seq, Dim)
            expert_outputs = [expert(x) for expert in self.experts] 
            # Stack 后 shape: (B, Num_Experts, Seq, Dim)
            expert_outputs = torch.stack(expert_outputs, dim=1) 
            
            # Weighted Sum
            
            B, N = self.routing_weights.shape
            view_shape = (B, N) + (1,) * (expert_outputs.ndim - 2)
            w = self.routing_weights.view(view_shape)
            
            # 现在 w 的形状例如是 (B, 3, 1, 1)，可以正确广播到 (B, 3, 4096, 768)
            adapter_out = torch.sum(w * expert_outputs, dim=1)
            
            return base_out + adapter_out