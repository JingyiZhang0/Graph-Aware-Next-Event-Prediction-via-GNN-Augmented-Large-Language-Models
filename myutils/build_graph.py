# Title: build_graph.py
# Author: Jingyi Zhang
# Date: 2025-12-12
# Filename: myutils/build_graph.py
# Description: Build activity transition graph with time and frequency edge attributes.

import torch
import networkx as nx
from torch_geometric.data import Data
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, Any


def build_activity_graph(
    df: pd.DataFrame,
    act_to_idx: Dict[str, int],
    config: Dict[str, Any] | None = None,
) -> Data:
    """Build activity graph with time deltas and transition frequencies.

    Args:
        df (pandas.DataFrame): DataFrame containing columns `case_id`, `activity`, `timestamp`.
        act_to_idx (dict): Mapping from label activity name (str) to index (int).
        config (dict, optional): Optional configuration dict (unused here).

    Returns:
        torch_geometric.data.Data: Graph data with fields:
            - edge_index (torch.Tensor): [2, E] source/target indices.
            - edge_attr (torch.Tensor): [E, 1] time delta (seconds).
            - edge_weight (torch.Tensor): [E, 1] transition frequency in [0,1].
            - num_nodes (int): Number of nodes in graph.
    """
    # Copy index dictionary to handle activities not in the label set
    graph_act_to_idx = {}
    all_activities = df['activity'].unique()
    next_idx = max(act_to_idx.values()) + 1 if act_to_idx else 0
    
    for act in all_activities:
        if act in act_to_idx:
            graph_act_to_idx[act] = act_to_idx[act]
        else:
            graph_act_to_idx[act] = next_idx
            next_idx += 1
    
    # Extract activity sequences for all cases
    cases = df.groupby('case_id')
    
    # Edge source nodes, target nodes, and relationship time intervals
    src_nodes = []
    dst_nodes = []
    edge_times = []
    
    # Collect all activity transition pairs for frequency calculation
    transition_counts = {}
    
    for _, group in cases:
        activities = group['activity'].values
        timestamps = group['timestamp'].values
        
        # Construct edges (activity transitions)
        for i in range(len(activities) - 1):
            src = graph_act_to_idx[activities[i]]
            dst = graph_act_to_idx[activities[i+1]]
            
            # Calculate time interval
            try:
                time_diff = (timestamps[i+1] - timestamps[i]).total_seconds()
            except AttributeError:
                time_diff = (timestamps[i+1] - timestamps[i]) / np.timedelta64(1, 's')
            
            src_nodes.append(src)
            dst_nodes.append(dst)
            edge_times.append(time_diff)
            
            # Record transition pairs for frequency calculation
            key = (src, dst)
            transition_counts[key] = transition_counts.get(key, 0) + 1
    
    # Calculate frequency
    total_transitions = len(src_nodes)
    edge_weights = [transition_counts[(src, dst)] / total_transitions 
                   for src, dst in zip(src_nodes, dst_nodes)]
    
    # Create edge attributes
    edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)
    edge_attr = torch.tensor(edge_times, dtype=torch.float).view(-1, 1)  # Time difference
    edge_weight = torch.tensor(edge_weights, dtype=torch.float).view(-1, 1)  # Frequency
    num_nodes = len(graph_act_to_idx)
    
    # Create graph data object
    data = Data(
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_weight=edge_weight,  # Add frequency as edge weight
        num_nodes=num_nodes
    )
    # attach mapping for later use
    data.act_to_idx_map = {act: idx for act, idx in graph_act_to_idx.items()}
    
    print(f"Constructed activity graph: {data.num_nodes} nodes, {data.num_edges} edges")
    
    return data