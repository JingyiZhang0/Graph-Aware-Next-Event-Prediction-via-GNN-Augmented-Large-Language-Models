# Title: train_gat.py
# Author: Jingyi Zhang
# Date: 2025-12-12
# Filename: train_gat.py
# Description: Train a GAT-based encoder with MLP classifier for next-activity prediction.
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from datetime import datetime

# Use your existing dataset/dataloader and GNN encoder
from torch.utils.data import DataLoader
from myutils.dataset import ActivityDataset, collate_fn, load_and_split_data
from myutils.build_graph import build_activity_graph
from myutils.config_loader import load_config
from models.gnn_encoder import TimeAwareGAT
import argparse


class SequenceMLPClassifier(nn.Module):
    """A simple MLP classifier on top of pooled GNN sequence embeddings."""
    def __init__(self, in_dim: int, num_labels: int, hidden: int = 256, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Dropout(dropout),
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_labels)
        )

    def forward(self, x):
        return self.net(x)


def pool_sequences_mean(gnn_node_emb: torch.Tensor, seqs: torch.Tensor, timestamps) -> torch.Tensor:
    """
    Mean-pool node embeddings for each sequence, masking out paddings using timestamps length.
    Args:
        gnn_node_emb: [num_nodes, gnn_dim] node embeddings from TimeAwareGAT
        seqs: [bs, T] padded activity index sequences (padding value=0 by collate_fn)
        timestamps: list of length bs, each is a list of true timestamps (length equals true seq length)
    Returns:
        [bs, gnn_dim] pooled sequence embeddings
    """
    bs, T = seqs.size()
    device = gnn_node_emb.device
    gnn_dim = gnn_node_emb.size(-1)
    pooled = []
    for i in range(bs):
        true_len = len(timestamps[i]) if timestamps is not None and timestamps[i] is not None else T
        true_len = max(1, min(true_len, T))  # safety clamp
        idx = seqs[i, :true_len]
        # Gather node embeddings and mean pool
        emb = gnn_node_emb[idx]
        pooled.append(emb.mean(dim=0))
    return torch.stack(pooled, dim=0)


def pool_sequences_recency(gnn_node_emb: torch.Tensor, seqs: torch.Tensor, timestamps, gamma: float = 0.9) -> torch.Tensor:
    """
    Recency-weighted average pooling: more recent steps get higher weights.
    If timestamps are provided, weight by position (most recent highest) to avoid time parsing overhead.
    Args:
        gnn_node_emb: [num_nodes, gnn_dim]
        seqs: [bs, T] indices into node_emb
        timestamps: list of per-seq timestamps; used only to get true lengths
        gamma: decay factor in (0,1); weight_t = gamma^(k) where k=steps-from-end
    Returns:
        [bs, gnn_dim]
    """
    bs, T = seqs.size()
    device = gnn_node_emb.device
    pooled = []
    for i in range(bs):
        true_len = len(timestamps[i]) if timestamps is not None and timestamps[i] is not None else T
        true_len = max(1, min(true_len, T))
        idx = seqs[i, :true_len]
        emb = gnn_node_emb[idx]  # [L, D]
        L = emb.size(0)
        # weights: newer steps larger; k=0 for last step
        exponents = torch.arange(L-1, -1, -1, device=emb.device, dtype=emb.dtype)
        weights = (gamma ** exponents).unsqueeze(1)  # [L,1]
        weights = weights / (weights.sum() + 1e-8)
        pooled.append((emb * weights).sum(dim=0))
    return torch.stack(pooled, dim=0)


def run_for_config(config_path: str):
    # 1) Load config and device
    config = load_config(path=config_path)
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

    # 2) Load raw data and parse timestamps (robust to mixed formats)
    df = pd.read_csv(config['data']['path'])
    try:
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='ISO8601')
    except ValueError:
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')

    # 3) Build train/val/test datasets from sliding windows
    datasets = load_and_split_data(df, window_size=config['data']['window_size'])
    act_to_idx = datasets['act_to_idx']
    num_labels = datasets['num_labels']
    config['model']['num_labels'] = num_labels

    train_ds = datasets['train']
    val_ds = datasets['val']
    test_ds = datasets['test']
    train_loader = DataLoader(train_ds, batch_size=config['data']['batch_size'], shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=config['data']['batch_size'], shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=config['data']['batch_size'], shuffle=False, collate_fn=collate_fn)

    # 4) Build global activity graph once
    graph_data = build_activity_graph(df, act_to_idx, config=config).to(device)
    # Optionally make graph undirected to improve message passing coverage
    if bool(config.get('model', {}).get('undirected', True)) and graph_data.edge_index.numel() > 0:
        ei = graph_data.edge_index
        ea = graph_data.edge_attr
        ew = getattr(graph_data, 'edge_weight', None)
        ei_rev = torch.stack([ei[1], ei[0]], dim=0)
        graph_data.edge_index = torch.cat([ei, ei_rev], dim=1)
        graph_data.edge_attr = torch.cat([ea, ea], dim=0)
        if ew is not None:
            graph_data.edge_weight = torch.cat([ew, ew], dim=0)
    # Edge time normalization controls
    if hasattr(graph_data, 'edge_attr') and graph_data.edge_attr is not None and graph_data.edge_attr.numel() > 0:
        edge_time = graph_data.edge_attr.view(-1)
        if bool(config.get('model', {}).get('log_edge_time', True)):
            edge_time = torch.log1p(torch.clamp(edge_time, min=0))
        if bool(config.get('model', {}).get('normalize_edge_time', True)):
            mean = edge_time.mean(); std = edge_time.std().clamp(min=1e-6)
            edge_time = (edge_time - mean) / std
        graph_data.edge_attr = edge_time.view(-1, 1)
    # Build a mapping from dataset indices -> graph indices using activity names
    # datasets['act_to_idx'] maps activity name -> label-space index (labels subset)
    # graph_data.act_to_idx_map maps activity name -> graph node index (all activities in df)
    ds_act_to_idx = datasets['act_to_idx']
    graph_act_to_idx = getattr(graph_data, 'act_to_idx_map', None)
    if graph_act_to_idx is None:
        raise RuntimeError("Graph does not carry act_to_idx_map; please rebuild graph with updated build_graph.py")
    # Build idx->act maps for each split's sequence index space
    idx_to_act_seq_train = {v: k for k, v in getattr(train_ds, 'seq_act_to_idx', {}).items()}
    idx_to_act_seq_val = {v: k for k, v in getattr(val_ds, 'seq_act_to_idx', {}).items()}
    idx_to_act_seq_test = {v: k for k, v in getattr(test_ds, 'seq_act_to_idx', {}).items()}

    # Quick coverage sanity checks
    gnames = set(graph_act_to_idx.keys())
    train_names = set(idx_to_act_seq_train.values())
    val_names = set(idx_to_act_seq_val.values())
    test_names = set(idx_to_act_seq_test.values())
    miss_train = train_names - gnames
    miss_val = val_names - gnames
    miss_test = test_names - gnames
    if miss_train or miss_val or miss_test:
        print("[Sanity] Activities in sequences but not in graph:")
        if miss_train: print(f"  Train missing: {len(miss_train)} examples like: {list(sorted(miss_train))[:5]}")
        if miss_val: print(f"  Val missing: {len(miss_val)} examples like: {list(sorted(miss_val))[:5]}")
        if miss_test: print(f"  Test missing: {len(miss_test)} examples like: {list(sorted(miss_test))[:5]}")

    def remap_batch_sequences_to_graph(seqs: torch.Tensor, idx_to_act_seq_map: dict) -> torch.Tensor:
        """Map dataset sequence indices (seq_act_to_idx space) to graph node indices by activity name."""
        seqs_cpu = seqs.detach().cpu()
        remapped = []
        for row in seqs_cpu:
            mapped_row = []
            for idx in row.tolist():
                # padding value 0 may be a valid index; we'll mask paddings using timestamps during pooling
                act_name = idx_to_act_seq_map.get(int(idx))
                if act_name is None:
                    # Unknown token index; default to node 0 to avoid OOB
                    mapped_row.append(0)
                else:
                    mapped_row.append(int(graph_act_to_idx.get(act_name, 0)))
            remapped.append(mapped_row)
        return torch.tensor(remapped, device=seqs.device, dtype=torch.long)

    # 5) Initialize learnable node embeddings, GNN encoder and MLP classifier
    gnn = TimeAwareGAT(config=config).to(device)
    gnn_dim = config['model']['gnn_embed_dim']
    # Node embeddings for all graph nodes
    node_embed_table = nn.Embedding(graph_data.num_nodes, gnn_dim).to(device)
    nn.init.xavier_uniform_(node_embed_table.weight)
    cls_hidden = int(config['model'].get('mlp_hidden', max(256, gnn_dim)))
    classifier = SequenceMLPClassifier(in_dim=gnn_dim, num_labels=num_labels, hidden=cls_hidden, dropout=0.3).to(device)

    # 6) Optimizer
    # More conservative defaults to avoid early instability
    lr_gnn = float(config['training'].get('gnn_learning_rate', 1e-4))
    lr_cls = float(config['training'].get('mlp_learning_rate', lr_gnn))
    lr_emb = float(config['training'].get('emb_learning_rate', lr_gnn))

    optimizer = torch.optim.Adam([
        {'params': node_embed_table.parameters(), 'lr': lr_emb},
        {'params': gnn.parameters(), 'lr': lr_gnn},
        {'params': classifier.parameters(), 'lr': lr_cls}
    ], weight_decay=float(config['training'].get('weight_decay', 1e-5)))

    num_epochs = int(config['training'].get('num_epochs', 10))
    best_val_f1 = 0.0
    best_state = None

    from sklearn.metrics import f1_score, accuracy_score
    # Class weights for imbalanced labels (computed from training dataset)
    from collections import Counter
    train_label_counts = Counter([lbl for _, lbl, _ in train_loader.dataset]) if hasattr(train_loader, 'dataset') else None
    use_class_weights = bool(config.get('training', {}).get('use_class_weights', True))
    if use_class_weights and train_label_counts:
        total = sum(train_label_counts.values())
        # Inverse frequency
        weights = torch.zeros(num_labels, dtype=torch.float, device=device)
        for c, cnt in train_label_counts.items():
            weights[int(c)] = total / max(1, cnt)
        # Normalize weights to mean=1 to keep loss scale stable
        weights = weights * (num_labels / weights.sum().clamp(min=1e-6))
    else:
        weights = None

    # Print an alignment sanity sample (sequence names vs label name)
    def idx_to_label_name(lbl_idx: int) -> str:
        # Map label index back to activity name using datasets['idx_to_act']
        return datasets.get('idx_to_act', {}).get(int(lbl_idx), str(lbl_idx))
    # Peek one batch to verify mapping and counts
    try:
        sample_seqs, sample_labels, sample_ts = next(iter(train_loader))
        sample_seqs_graph = None
        if len(sample_seqs) > 0:
            sample_seqs_graph = remap_batch_sequences_to_graph(sample_seqs, idx_to_act_seq_train)
            # Unknown ratio in first batch
            unknown_ratio = (sample_seqs_graph == 0).float().mean().item() if sample_seqs_graph.numel() > 0 else 0.0
            print(f"[Debug] First-batch remap unknown-as-0 ratio: {unknown_ratio:.3f}")
            # Print first sample names
            row = sample_seqs[0].tolist()
            row_names = [idx_to_act_seq_train.get(int(i), "?") for i in row]
            print(f"[Debug] First seq names (raw, padded): {row_names[:15]}")
            print(f"[Debug] First label name: {idx_to_label_name(sample_labels[0].item())}")
    except StopIteration:
        pass

    for epoch in range(1, num_epochs + 1):
        # ===== Train =====
        gnn.train(); classifier.train()
        total_loss, total_acc, total_batches = 0.0, 0.0, 0
        for bidx, (seqs, labels, timestamps) in enumerate(train_loader):
            seqs = seqs.to(device)
            labels = labels.to(device)

            # Compute node embeddings for the static graph
            graph_data.x = node_embed_table.weight  # feed learnable node features
            node_emb = gnn(graph_data)  # [num_nodes, gnn_dim]
            if bidx == 0:
                # Node embedding variance diagnostic
                with torch.no_grad():
                    var_all = node_emb.var(dim=0).mean().item()
                    var_per_node = node_emb.var(dim=1).mean().item()
                    print(f"[Debug] node_emb var(mean over dims)={var_all:.6f}, var(mean over nodes)={var_per_node:.6f}")

            # Remap sequence indices to graph node indices, then mean-pool (mask paddings using timestamps)
            seqs_graph = remap_batch_sequences_to_graph(seqs, idx_to_act_seq_train)
            pooling_mode = str(config.get('training', {}).get('pooling', 'recency')).lower()
            if pooling_mode == 'recency':
                gamma = float(config.get('training', {}).get('decay_gamma', 0.9))
                seq_emb = pool_sequences_recency(node_emb, seqs_graph, timestamps, gamma)
            elif pooling_mode == 'last':
                # Take the last valid step embedding
                bs, T = seqs_graph.size()
                pooled = []
                for i in range(bs):
                    L = len(timestamps[i]) if timestamps[i] is not None else T
                    L = max(1, min(L, T))
                    last_idx = seqs_graph[i, L-1]
                    pooled.append(node_emb[last_idx])
                seq_emb = torch.stack(pooled, dim=0)
            else:
                seq_emb = pool_sequences_mean(node_emb, seqs_graph, timestamps)  # [bs, gnn_dim]

            logits = classifier(seq_emb)
            loss = F.cross_entropy(logits, labels, weight=weights)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([*gnn.parameters(), *classifier.parameters()], max_norm=1.0)
            optimizer.step()

            preds = logits.argmax(dim=1)
            acc = (preds == labels).float().mean().item()
            total_loss += loss.item(); total_acc += acc; total_batches += 1

        avg_train_loss = total_loss / max(1, total_batches)
        avg_train_acc = total_acc / max(1, total_batches)

        # ===== Validate =====
        gnn.eval(); classifier.eval()
        val_losses, val_preds, val_trues = [], [], []
        with torch.no_grad():
            for seqs, labels, timestamps in val_loader:
                seqs = seqs.to(device)
                labels = labels.to(device)
                graph_data.x = node_embed_table.weight
                node_emb = gnn(graph_data)
                seqs_graph = remap_batch_sequences_to_graph(seqs, idx_to_act_seq_val)
                pooling_mode = str(config.get('training', {}).get('pooling', 'recency')).lower()
                if pooling_mode == 'recency':
                    gamma = float(config.get('training', {}).get('decay_gamma', 0.9))
                    seq_emb = pool_sequences_recency(node_emb, seqs_graph, timestamps, gamma)
                elif pooling_mode == 'last':
                    bs, T = seqs_graph.size()
                    pooled = []
                    for i in range(bs):
                        L = len(timestamps[i]) if timestamps[i] is not None else T
                        L = max(1, min(L, T))
                        last_idx = seqs_graph[i, L-1]
                        pooled.append(node_emb[last_idx])
                    seq_emb = torch.stack(pooled, dim=0)
                else:
                    seq_emb = pool_sequences_mean(node_emb, seqs_graph, timestamps)
                logits = classifier(seq_emb)
                loss = F.cross_entropy(logits, labels, weight=weights)
                preds = logits.argmax(dim=1)
                val_losses.append(loss.item())
                val_preds.extend(preds.detach().cpu().tolist())
                val_trues.extend(labels.detach().cpu().tolist())

        val_acc = accuracy_score(val_trues, val_preds) if val_trues else 0.0
        val_f1 = f1_score(val_trues, val_preds, average='weighted', zero_division=0) if val_trues else 0.0

        print(f"Epoch {epoch:02d} | TrainLoss {avg_train_loss:.4f} Acc {avg_train_acc:.4f} | ValAcc {val_acc:.4f} F1 {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {
                'gnn': gnn.state_dict(),
                'classifier': classifier.state_dict(),
                'epoch': epoch,
                'val_f1': best_val_f1
            }

    # Restore best
    if best_state is not None:
        gnn.load_state_dict(best_state['gnn'])
        classifier.load_state_dict(best_state['classifier'])
        print(f"Loaded best model at epoch {best_state['epoch']} with ValF1={best_state['val_f1']:.4f}")

    # ===== Test =====
    gnn.eval(); classifier.eval()
    from sklearn.metrics import classification_report, precision_score, recall_score
    test_preds, test_trues = [], []
    test_losses, test_accs = [], []
    with torch.no_grad():
        for seqs, labels, timestamps in test_loader:
            seqs = seqs.to(device)
            labels = labels.to(device)
            graph_data.x = node_embed_table.weight
            node_emb = gnn(graph_data)
            seqs_graph = remap_batch_sequences_to_graph(seqs, idx_to_act_seq_test)
            pooling_mode = str(config.get('training', {}).get('pooling', 'recency')).lower()
            if pooling_mode == 'recency':
                gamma = float(config.get('training', {}).get('decay_gamma', 0.9))
                seq_emb = pool_sequences_recency(node_emb, seqs_graph, timestamps, gamma)
            elif pooling_mode == 'last':
                bs, T = seqs_graph.size()
                pooled = []
                for i in range(bs):
                    L = len(timestamps[i]) if timestamps[i] is not None else T
                    L = max(1, min(L, T))
                    last_idx = seqs_graph[i, L-1]
                    pooled.append(node_emb[last_idx])
                seq_emb = torch.stack(pooled, dim=0)
            else:
                seq_emb = pool_sequences_mean(node_emb, seqs_graph, timestamps)
            logits = classifier(seq_emb)
            loss = F.cross_entropy(logits, labels, weight=weights)
            preds = logits.argmax(dim=1)
            test_losses.append(loss.item())
            test_accs.append((preds == labels).float().mean().item())
            test_preds.extend(preds.detach().cpu().tolist())
            test_trues.extend(labels.detach().cpu().tolist())

    from numpy import mean
    print(f"\n=== Test Results ({config_path}) ===")
    print(f"Test Loss: {mean(test_losses):.4f} | Test Acc: {mean(test_accs):.4f}")
    print(f"Test F1: {f1_score(test_trues, test_preds, average='weighted', zero_division=0):.4f}")
    print(f"Test Precision: {precision_score(test_trues, test_preds, average='weighted', zero_division=0):.4f}")
    print(f"Test Recall: {recall_score(test_trues, test_preds, average='weighted', zero_division=0):.4f}")

    # Optional per-class report
    # You can map indices back to activity names if needed using datasets['idx_to_act']
    # print(classification_report(test_trues, test_preds))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/BPIC2012.yaml", help="Path to a single YAML config")
    args = parser.parse_args()

    run_for_config(args.config)