from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class FeRAConfig:
    rank: int = 4
    alpha: float = 1.0  # scaling factor = alpha / rank
    dropout: float = 0.0
    
    num_bands: int = 3     
    num_experts: int = 3    
    router_tau: float = 0.7 
    
    target_modules: List[str] = field(default_factory=lambda: ["to_q", "to_k", "to_v", "to_out.0"])
    
    fecl_weight: float = 0.1