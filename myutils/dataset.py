# Title: dataset.py
# Author: Jingyi Zhang
# Date: 2025-12-12
# Filename: myutils/dataset.py
# Description: Dataset utilities for activity sequence prediction with timestamps.

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from typing import List, Dict, Union, Tuple, Optional
from datetime import datetime

class ActivityDataset(Dataset):
    """
    Activity prediction dataset class.

    Args:
        sequences (List[List[str]]): List of activity sequences (each sequence is a list of activity names).
        labels (List[str]): Corresponding next-activity labels for each sequence.
        act_to_idx (Dict[str, int]): Mapping from label activity name to index.
        timestamps (Optional[List[List[datetime]]]): Per-sequence timestamp lists (same length as sequences).

    Returns:
        Tuple[torch.Tensor, torch.Tensor, Optional[List[datetime]]]: Sequence indices, label index, and timestamps list per item.
    """
    def __init__(
        self,
        sequences: List[List[str]],
        labels: List[str],
        act_to_idx: Dict[str, int],
        timestamps: Optional[List[List[datetime]]] = None,
    ) -> None:
        self.sequences = sequences
        self.labels = labels
        self.act_to_idx = act_to_idx
        
        # Create an extended activity index dictionary, including all activities appearing in the sequences
        self.seq_act_to_idx = dict(act_to_idx)  # Copy the label index dictionary
        next_idx = max(act_to_idx.values()) + 1 if act_to_idx else 0
        
        # Create new indices for activities appearing in the sequences but not in the label set
        for seq in sequences:
            for act in seq:
                if act not in self.seq_act_to_idx:
                    self.seq_act_to_idx[act] = next_idx
                    next_idx += 1
        
        if timestamps is not None:
            assert len(timestamps) == len(sequences), "Timestamps length must match sequences"
        self.timestamps = timestamps if timestamps is not None else [None] * len(sequences)
        
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Optional[List[datetime]]]:
        seq = self.sequences[idx]
        if isinstance(seq, np.ndarray):
            seq = seq.tolist()

        # Process the sequence using the extended activity index dictionary
        seq_idx = [self.seq_act_to_idx[act] for act in seq]
        label_idx = self.act_to_idx[self.labels[idx]]  # Labels must be in the original act_to_idx
        ts_seq = self.timestamps[idx]
        return torch.tensor(seq_idx), torch.tensor(label_idx), ts_seq

def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor, Optional[List[datetime]]]]) -> Tuple[torch.Tensor, torch.Tensor, List[Optional[List[datetime]]]]:
    """
    Batch collate function that supports timestamps.
    Args:
        batch (List[Tuple[torch.Tensor, torch.Tensor, Optional[List[datetime]]]]): Items from ActivityDataset.
    Returns:
        padded_sequences (torch.Tensor): Shape [B, T], padded sequence indices.
        labels (torch.Tensor): Shape [B], label indices.
        timestamps (List[Optional[List[datetime]]]): Per-item timestamps lists.
    """
    sequences, labels, timestamps = zip(*batch)

    padded_seqs = torch.nn.utils.rnn.pad_sequence(
        sequences, batch_first=True, padding_value=0
    )
    labels = torch.stack(labels)
    return padded_seqs, labels, list(timestamps)


def load_and_split_data(
    df: pd.DataFrame,
    window_size: int = 3,
    test_ratio: float = 0.2,
    val_ratio: float = 0.1,
    min_samples_per_label: int = 3,
) -> Dict[str, Union[ActivityDataset, Dict[str, int], int, List[str], Dict[str, List[List[datetime]]]]]:
    """
    Load and split time series data with stratified sampling for better label distribution.

    Args:
        df (pandas.DataFrame): Input dataframe with columns `case_id`, `activity`, `timestamp`.
        window_size (int): Number of prefix steps for sequence windows.
        test_ratio (float): Fraction of samples per label placed into test split.
        val_ratio (float): Fraction of samples per label placed into validation split.
        min_samples_per_label (int): Minimum samples required for a label to be kept.

    Returns:
        Dict[str, Any]: A dict containing datasets and metadata keys:
            - 'train', 'val', 'test' (ActivityDataset): Dataset objects.
            - 'act_to_idx' (Dict[str,int]): Label mapping.
            - 'idx_to_act' (Dict[int,str]): Reverse label mapping.
            - 'num_labels' (int): Number of labels.
            - 'all_activities' (List[str]): All activities in df.
            - 'label_activities' (List[str]): Valid labels after filtering.
            - 'timestamps' (Dict[str, List[List[datetime]]]): Per-split timestamps.
    """
    df = df.sort_values(['case_id', 'timestamp'])
    
    # Get the list of all activities (for input sequences)
    all_activities = df['activity'].unique().tolist()
    
    # Collect all possible labels and samples from the entire dataset first
    all_samples = []  # Store (case_id, sequence, label, timestamps)
    all_label_activities = set()
    
    for case_id, group in df.groupby('case_id'):
        acts = group['activity'].tolist()
        times = group['timestamp'].values
        if len(acts) <= window_size:
            continue
        for i in range(len(acts) - window_size):
            next_act = acts[i+window_size]
            all_label_activities.add(next_act)
            all_samples.append({
                'case_id': case_id,
                'sequence': acts[i:i+window_size],
                'label': next_act,
                'timestamps': [pd.Timestamp(t).to_pydatetime() for t in times[i:i+window_size]]
            })
    
    # Filter out labels with sample size less than min_samples_per_label
    from collections import Counter
    label_counts = Counter([sample['label'] for sample in all_samples])
    valid_labels = {label for label, count in label_counts.items() if count >= min_samples_per_label}
    
    print(f"Total number of labels: {len(all_label_activities)}")
    print(f"Number of labels with samples >= {min_samples_per_label}: {len(valid_labels)}")
    if len(all_label_activities) - len(valid_labels) > 0:
        removed_labels = all_label_activities - valid_labels
        print(f"Removed labels (samples < {min_samples_per_label}): {removed_labels}")

    # Filter samples, keeping only those with valid labels
    valid_samples = [sample for sample in all_samples if sample['label'] in valid_labels]

    # Create a globally stable label mapping
    label_activities = sorted(list(valid_labels))
    act_to_idx = {act: i for i, act in enumerate(label_activities)}
    idx_to_act = {i: act for i, act in enumerate(label_activities)}

    # Stratified sampling: ensure reasonable distribution of each label in training, validation, and test sets
    import random
    random.seed(42)  # Set random seed for reproducibility

    train_samples, val_samples, test_samples = [], [], []

    for label in label_activities:
        # Get all samples for the label
        label_samples = [s for s in valid_samples if s['label'] == label]
        random.shuffle(label_samples)  # Shuffle randomly

        n_samples = len(label_samples)
        n_test = max(1, int(n_samples * test_ratio))  # At least 1 sample
        n_val = max(1, int(n_samples * val_ratio))    # At least 1 sample
        n_train = n_samples - n_test - n_val
        
        # Ensure at least 1 sample in the training set
        if n_train < 1:
            n_train = 1
            n_test = max(0, (n_samples - n_train) // 2)
            n_val = n_samples - n_train - n_test
        
        # Allocate samples
        test_samples.extend(label_samples[:n_test])
        val_samples.extend(label_samples[n_test:n_test+n_val])
        train_samples.extend(label_samples[n_test+n_val:])

    # Extract sequences, labels, and timestamps
    def extract_data(samples):
        sequences = [s['sequence'] for s in samples]
        labels = [s['label'] for s in samples]
        timestamps = [s['timestamps'] for s in samples]
        return sequences, labels, timestamps
    
    train_seqs, train_labels, train_times = extract_data(train_samples)
    val_seqs, val_labels, val_times = extract_data(val_samples)
    test_seqs, test_labels, test_times = extract_data(test_samples)
    
    # Print statistics
    print(f"\nTotal number of activities: {len(all_activities)}")
    print(f"Number of valid labels: {len(label_activities)}")
    print(f"Activities not appearing as labels: {set(all_activities) - set(label_activities)}")
    
    # Detailed label distribution statistics
    train_label_counts = Counter(train_labels)
    val_label_counts = Counter(val_labels)
    test_label_counts = Counter(test_labels)

    print(f"\n=== Detailed Label Distribution Statistics ===")
    print(f"Training set label distribution: {len(set(train_labels))} / {len(label_activities)}")
    print(f"Validation set label distribution: {len(set(val_labels))} / {len(label_activities)}")
    print(f"Test set label distribution: {len(set(test_labels))} / {len(label_activities)}")

    print(f"\n=== Number of Samples per Label ===")
    print(f"{'Label Name':<40} {'Train':<8} {'Validation':<8} {'Test':<8} {'Total':<8} {'Train%':<8} {'Validation%':<8} {'Test%':<8}")
    print("-" * 120)

    for label in label_activities:
        train_count = train_label_counts.get(label, 0)
        val_count = val_label_counts.get(label, 0)
        test_count = test_label_counts.get(label, 0)
        total_count = train_count + val_count + test_count
        
        train_pct = f"{train_count/total_count*100:.1f}%" if total_count > 0 else "0%"
        val_pct = f"{val_count/total_count*100:.1f}%" if total_count > 0 else "0%"
        test_pct = f"{test_count/total_count*100:.1f}%" if total_count > 0 else "0%"
        
        print(f"{label:<40} {train_count:<8} {val_count:<8} {test_count:<8} {total_count:<8} {train_pct:<8} {val_pct:<8} {test_pct:<8}")

    print("-" * 120)
    print(f"{'Total':<40} {len(train_labels):<8} {len(val_labels):<8} {len(test_labels):<8} {len(train_labels) + len(val_labels) + len(test_labels):<8}")

    # Check if all labels appear in each dataset
    missing_in_train = set(label_activities) - set(train_labels)
    missing_in_val = set(label_activities) - set(val_labels)
    missing_in_test = set(label_activities) - set(test_labels)

    if missing_in_train:
        print(f"\n⚠️  Labels missing in training set: {missing_in_train}")
    if missing_in_val:
        print(f"⚠️  Labels missing in validation set: {missing_in_val}")
    if missing_in_test:
        print(f"⚠️  Labels missing in test set: {missing_in_test}")
    
    if not (missing_in_train or missing_in_val or missing_in_test):
        print(f"\n✅ All labels appear in training, validation, and test sets")

    return {
        'train': ActivityDataset(train_seqs, train_labels, act_to_idx, timestamps=train_times),
        'val': ActivityDataset(val_seqs, val_labels, act_to_idx, timestamps=val_times),
        'test': ActivityDataset(test_seqs, test_labels, act_to_idx, timestamps=test_times),
        'act_to_idx': act_to_idx,
        'idx_to_act': idx_to_act,
        'num_labels': len(label_activities),
        'all_activities': all_activities,
        'label_activities': label_activities,
        'timestamps': {
            'train': train_times,
            'val': val_times,
            'test': test_times
        }
    }