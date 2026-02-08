"""
Velocity field neural network for rectified flow distillation.
"""

import torch
import torch.nn as nn
from .embeddings import SinusoidalPosEmb


class VelocityField(nn.Module):
    """
    A neural network model for the velocity field v_theta(Z_t, t, j).
    """
    
    def __init__(self, d_input, d_model=768, num_distill_layers=6, n_layers=4):
        super().__init__()
        self.d_model = d_model  # use teacher dimension
        self.d_input = d_input
        self.layer_emb = nn.Embedding(num_distill_layers, d_model)
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.input_proj = nn.Linear(d_input, d_model)
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Linear(d_model * 4, d_model),
                nn.Dropout(p=0.1)
            ) for _ in range(n_layers)
        ])
        # self.ln_out = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, d_input)

    def forward(self, z_t, t, j):
        is_3d_input = z_t.dim() == 3
        if is_3d_input:
            B, L, V = z_t.shape
            z_t = z_t.reshape(B * L, V)
            t = t.unsqueeze(-1).expand(-1, L).reshape(B * L)
            j = j.unsqueeze(-1).expand(-1, L).reshape(B * L)
        
        t_emb = self.time_emb(t)
        j_emb = self.layer_emb(j)
        
        x = self.input_proj(z_t) + t_emb + j_emb
        
        for layer in self.layers:
            x = layer(x) + x
            
        # x = self.ln_out(x)
        x = self.output_proj(x)

        if is_3d_input:
            x = x.view(B, L, -1)
            
        return x