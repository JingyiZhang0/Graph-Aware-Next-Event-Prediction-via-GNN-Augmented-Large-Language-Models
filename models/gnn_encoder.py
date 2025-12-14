# Title: gnn_encoder.py
# Author: Jingyi Zhang
# Date: 2025-12-12
# Filename: models/gnn_encoder.py
# Description: Time-aware GNN encoder modules for activity graphs.

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from typing import Dict, Any
from torch_geometric.data import Data



class TimeEncoding(nn.Module):
    def __init__(self, time_dim: int):
        """TimeEncoding

        Encodes scalar time deltas into a learned embedding.

        Args:
            time_dim (int): Output embedding dimension for time.

        Attributes:
            linear (torch.nn.Linear): Linear projection from 1 -> time_dim.
            activation (torch.nn.ReLU): Non-linearity applied to projection.
        """
        super().__init__()
        self.linear = nn.Linear(1, time_dim)
        self.activation = nn.ReLU()

    def forward(self, time_deltas: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            time_deltas (torch.Tensor): Shape [E] or [E, 1], scalar time differences.

        Returns:
            torch.Tensor: Shape [E, time_dim], encoded time embeddings.
        """
        return self.activation(self.linear(time_deltas.unsqueeze(-1)))
    

class TimeAwareGAT(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        """TimeAwareGAT

        GAT-based encoder augmented with edge time embeddings and optional edge weights.

        Args:
            config (dict): Configuration dict with keys:
                - model.gnn_embed_dim (int): Node embedding dimension.
                - model.time_dim (int, optional): Time embedding dimension (default 16).

        Attributes:
            time_dim (int): Time embedding dimension used in the model.
            time_encoder (TimeEncoding): Module to encode edge time deltas.
            gat (torch_geometric.nn.GATConv): Graph attention layer supporting edge attributes.
        """
        super().__init__()
        in_channels = config['model']['gnn_embed_dim']
        out_channels = config['model']['gnn_embed_dim']
        self.time_dim = config.get('model', {}).get('time_dim', 16)  # Store time_dim in the class
        self.time_encoder = TimeEncoding(self.time_dim)
        self.gat = GATConv(
            in_channels + self.time_dim, 
            out_channels,
            edge_dim=1,  # New: support edge weights
            add_self_loops=False  # It is recommended to disable self-loops to avoid interference
        )

    def forward(self, data: Data) -> torch.Tensor:
        """Forward pass.

        Args:
            data (torch_geometric.data.Data): Graph data with fields:
                - x (torch.Tensor): Node features [N, in_dim] or None.
                - edge_index (torch.Tensor): Edge index [2, E].
                - edge_attr (torch.Tensor): Edge time deltas [E, 1].
                - edge_weight (torch.Tensor, optional): Edge frequency weights [E, 1].

        Returns:
            torch.Tensor: Node embeddings [N, out_channels].
        """
        x, edge_index, edge_time = data.x, data.edge_index, data.edge_attr
        edge_weight = getattr(data, 'edge_weight', None)
        
        # Fix: use feature dimensions that match GATConv input dimensions
        in_channels = self.gat.in_channels - self.time_dim  
        
        if x is None:
            x = torch.ones((data.num_nodes, in_channels), device=edge_index.device)
        elif x.size(1) != in_channels:
            # Handle dimension mismatch
            x = torch.ones((data.num_nodes, in_channels), device=edge_index.device)
    
        # Time embedding
        time_emb = self.time_encoder(edge_time)
        
        # Handle dimension issues
        if time_emb.dim() > 2:
            time_emb = time_emb.view(time_emb.size(0), -1)
    
        # The remaining code remains unchanged
        src_nodes = edge_index[0]
        x_time = torch.cat([x[src_nodes], time_emb], dim=1)
        x_extended = torch.zeros((x.size(0), x_time.size(1)), device=x.device)
        x_extended[src_nodes] = x_time
        
        # Use edge weights as edge attributes
        if edge_weight is not None:
            node_embeddings = self.gat(x_extended, edge_index, edge_attr=edge_weight)  # Use edge weights as edge attributes
        else:
            node_embeddings = self.gat(x_extended, edge_index, edge_attr=edge_time)
        
        return node_embeddings  # [num_nodes, out_channels]



