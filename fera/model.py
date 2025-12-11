import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import FrequencyEnergyIndicator
from .layer import FeRALinear
from .config import FeRAConfig

class SoftFrequencyRouter(nn.Module):
    def __init__(self, num_bands, num_experts, tau=0.7):
        super().__init__()
        self.tau = tau
        self.net = nn.Sequential(
            nn.Linear(num_bands, 64),
            nn.ReLU(),
            nn.Linear(64, num_experts)
        )
    
    def forward(self, e_t):
        logits = self.net(e_t)
        return F.softmax(logits / self.tau, dim=-1)

class FeRAModel(nn.Module):
    def __init__(self, base_model: nn.Module, config: FeRAConfig):
        super().__init__()
        self.config = config
        self.base_model = base_model
        
        self.fei_computer = FrequencyEnergyIndicator(num_bands=config.num_bands)
        self.router = SoftFrequencyRouter(config.num_bands, config.num_experts, config.router_tau)
        
        self.fera_layers = []
        self._inject_fera_layers(self.base_model)
        print(f"Successfully injected {len(self.fera_layers)} FeRA layers.")

    def _inject_fera_layers(self, module):
        for name, child in module.named_children():
            if isinstance(child, nn.Linear) and any(t in name for t in self.config.target_modules):
                fera_layer = FeRALinear(child, self.config)
                setattr(module, name, fera_layer)
                self.fera_layers.append(fera_layer)
            else:
                self._inject_fera_layers(child)

    def prepare_forward(self, z_t):
        self.current_e_t, _ = self.fei_computer(z_t) 
        
        weights = self.router(self.current_e_t) # (B, num_experts)
        
        for layer in self.fera_layers:
            layer.set_routing_weights(weights)
            
        return weights

    def compute_fecl_loss(self, z_base, z_fera, z_target):
        delta = z_fera - z_base
        residual = z_fera - z_target
        
        _, delta_bands = self.fei_computer(delta)
        _, resid_bands = self.fei_computer(residual)
        
        resid_energies = [torch.norm(b, p=2, dim=[1,2,3])**2 for b in resid_bands]
        total_energy = sum(resid_energies) + 1e-8
        weights = [e / total_energy for e in resid_energies]
        
        loss = 0.0
        for k in range(len(delta_bands)):
            norm_delta_k = torch.norm(delta_bands[k], p=2, dim=[1,2,3])
            norm_delta_total = torch.norm(delta, p=2, dim=[1,2,3]) + 1e-8
            
            norm_resid_k = torch.norm(resid_bands[k], p=2, dim=[1,2,3])
            norm_resid_total = torch.norm(residual, p=2, dim=[1,2,3]) + 1e-8
            
            # 论文 Eq. (10)
            term = (norm_delta_k / norm_delta_total - norm_resid_k / norm_resid_total) ** 2
            loss += weights[k] * term
            
        return loss.mean() * self.config.fecl_weight

    def save_adapters(self, path):
        state_dict = {}
        for k, v in self.router.state_dict().items():
            state_dict[f"router.{k}"] = v
            
        for name, param in self.base_model.named_parameters():
            if "experts" in name:
                state_dict[f"base_model.{name}"] = param
                
        torch.save(state_dict, path)
        print(f"Saved FeRA adapters to {path}")

    def load_adapters(self, path):
        state_dict = torch.load(path)
        router_dict = {k.replace("router.", ""): v for k, v in state_dict.items() if k.startswith("router.")}
        self.router.load_state_dict(router_dict)
        
        model_dict = {k.replace("base_model.", ""): v for k, v in state_dict.items() if k.startswith("base_model.")}
        self.base_model.load_state_dict(model_dict, strict=False)
        print("Loaded FeRA adapters.")