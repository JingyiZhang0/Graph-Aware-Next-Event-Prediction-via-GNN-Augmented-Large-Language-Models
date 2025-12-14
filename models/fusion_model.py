# Title: fusion_model.py
# Author: Jingyi Zhang
# Date: 2025-12-12
# Filename: models/fusion_model.py
# Description: Fusion layer combining GNN and text embeddings with cross-modal attention.

import torch
import torch.nn as nn
from typing import Dict, Any

class FusionLayer(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        """FusionLayer

        Multimodal fusion layer using cross-modal attention.

        Args:
            config (dict): Configuration dict with keys:
                - model.gnn_embed_dim (int): Dimension of GNN embedding.
                - model.text_embed_dim (int): Dimension of text embedding.
                - model.fusion_hidden_dim (int): Projected hidden dim (default 256).
                - model.fusion_num_heads (int): Attention heads (default 4).

        Attributes:
            gnn_proj (torch.nn.Linear): Linear projection for GNN embeddings.
            text_proj (torch.nn.Linear): Linear projection for text embeddings.
            cross_attn (torch.nn.MultiheadAttention): Cross-modal attention module.
        """
        super().__init__()
        gnn_dim = config['model']['gnn_embed_dim']
        text_dim = config['model']['text_embed_dim']
        hidden_dim = config['model'].get('fusion_hidden_dim', 256)
        num_heads = config['model'].get('fusion_num_heads', 4)

        self.gnn_proj = nn.Linear(gnn_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=False
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize projection weights using Xavier uniform and zero biases."""
        for proj in [self.gnn_proj, self.text_proj]:
            nn.init.xavier_uniform_(proj.weight)
            nn.init.zeros_(proj.bias)

    def forward(self, gnn_emb: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            gnn_emb (torch.Tensor): Shape [batch_size, gnn_dim], GNN sequence embedding.
            text_emb (torch.Tensor): Shape [batch_size, text_dim], text embedding.

        Returns:
            torch.Tensor: Concatenated embeddings of shape [batch_size, gnn_dim + text_dim].
        """
        return torch.cat([gnn_emb, text_emb], dim=1)