# Title: train_predict.py
# Author: Jingyi Zhang
# Date: 2025-12-12
# Filename: train_predict.py
# Description: Hybrid training script combining GNN encoder with LLM classifier and pure LLM baseline.
import torch
from torch.utils.data import DataLoader
from myutils.dataset import ActivityDataset, collate_fn
from myutils.text_processing import TextProcessor
#from models.gnn_encoder import EnhancedActivityGNN
from models.gnn_encoder import TimeAwareGAT
from models.fusion_model import FusionLayer
from models.llm_predictor import FusionLLMClassifier
import torch.nn.functional as F

#from models.llm_predictor import LLMPredictor
import gc
from torch.utils.tensorboard import SummaryWriter
from itertools import chain
from datetime import datetime
import os
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, AutoModel

# Main program usage
import pandas as pd
from myutils.dataset import load_and_split_data, ActivityDataset
from myutils.build_graph import build_activity_graph
from myutils.config_loader import load_config




# Load configuration
config = load_config(path = "configs/helpdesk.yaml")
use_gnn = config.get('model', {}).get('use_gnn', True)  # New flag: set to False for LLM-only mode
device = config['device']

# Load data
df = pd.read_csv(config['data']['path'])
# Handle mixed timestamp formats
try:
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='ISO8601')
except ValueError:
    # If ISO8601 fails, try mixed format inference
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')

data_path = config['data']['path'].lower()
if 'helpdesk' in data_path:
    source = 'helpdesk'
elif 'bpic2020' in data_path:
    source = 'bpic2020'
elif 'bpic2017' in data_path:
    source = 'bpic2017'
elif 'bpic2012' in data_path:
    source = 'bpic2012'
elif 'sepsis' in data_path:
    source = 'sepsis'
else:
    source = 'default'

# Data splitting
datasets = load_and_split_data(df, window_size=config['data']['window_size'])
config['model']['num_labels'] = len(datasets['act_to_idx'])  # Set number of labels

graph_data = None
if use_gnn:
    # Build graph only when GNN is enabled
    graph_data = build_activity_graph(df, datasets['act_to_idx'], config=config).to(device)

# Text preprocessing
text_processor = TextProcessor(source=source)
act_to_text = text_processor.clean_activity_names(list(datasets['act_to_idx'].keys()))

# DataLoader
train_set = datasets['train']
train_loader = DataLoader(train_set, batch_size=config['data']['batch_size'], shuffle=True, collate_fn=collate_fn)

# Initialize GNN model if needed
gnn = None
if use_gnn:
    gnn = TimeAwareGAT(config=config).to(device)

# Initialize LLM (base model + classification head)

print(f"[Info] Loading tokenizer and model: {config['model']['model_name']}")
tokenizer = AutoTokenizer.from_pretrained(config['model']['model_name'])
llm_backbone = AutoModel.from_pretrained(config['model']['model_name'])
print(f"[Info] Tokenizer class: {tokenizer.__class__.__name__}, fast={getattr(tokenizer, 'is_fast', False)}")
print(f"[Info] Model class: {llm_backbone.__class__.__name__}")

# Optionally freeze backbone params (model-agnostic)
if config['model'].get('freeze_bert', False):
    for p in llm_backbone.parameters():
        p.requires_grad = False
    print("[Info] Backbone parameters are frozen (freeze_bert=True)")
else:
    print("[Info] Backbone parameters are trainable (freeze_bert=False)")

# Get dimension configurations
gnn_dim = config['model']['gnn_embed_dim']
llm_embed_dim = llm_backbone.config.hidden_size
print(f"[Info] Model hidden size: {llm_embed_dim}")
num_labels = config['model']['num_labels']

# Fusion + classification model
llm = FusionLLMClassifier(
    llm_model=llm_backbone,
    gnn_emb_dim=gnn_dim,
    llm_embed_dim=llm_embed_dim,
    num_labels=num_labels,
    dropout=0.3  # Add dropout
).to(device)

# Optimizer
trainable_params = []
if use_gnn:
    trainable_params.append({'params': gnn.parameters(), 'lr': float(config['training']['gnn_learning_rate'])})
trainable_params.append({'params': llm.parameters(), 'lr': float(config['training']['llm_learning_rate'])})
optimizer = torch.optim.Adam(trainable_params, weight_decay=1e-5)

# TensorBoard logging setup
exp_id = 1
log_base = config['logging']['log_base']
while os.path.exists(f"{log_base}_{exp_id}"):
    exp_id += 1
log_dir = f"{log_base}_{exp_id}"
writer = SummaryWriter(log_dir=log_dir)
global_step = 0

# Create complete activity mapping, using only label activities
idx_to_act_all = {}
for act, idx in datasets['act_to_idx'].items():
    idx_to_act_all[idx] = act

for epoch in range(config['training']['num_epochs']): 
    if use_gnn:
        gnn.train()
    llm.train()

    total_loss = 0
    total_acc = 0
    total_batches = 0
    sample_logged = False

    for batch_idx, (seqs, labels, timestamps) in enumerate(train_loader):
        seqs, labels = seqs.to(device), labels.to(device)

    # Construct text input
        seq_texts = []
        for seq, ts_list in zip(seqs.cpu().tolist(), timestamps):
            acts = [idx_to_act_all.get(i, f"Unknown_Activity_{i}") for i in seq]
            ts_list = [pd.Timestamp(t).to_pydatetime() if not isinstance(t, datetime) else t for t in ts_list]
            sentence = text_processor.sequence_to_sentence_with_time(acts, ts_list, act_to_text)
            seq_texts.append(sentence)

        # Tokenizer encoding into token ids
        inputs = tokenizer(seq_texts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)

        input_ids, attention_mask = inputs["input_ids"].to(device), inputs["attention_mask"].to(device)

        if use_gnn:
            # GNN embedding per full graph
            gnn_emb_full = gnn(graph_data)  # [num_nodes, gnn_dim]
            # Get token embeddings via model-agnostic API
            with torch.no_grad():
                token_embs = llm.llm.get_input_embeddings()(input_ids)
            # Aggregate sequence gnn embeddings (mean pooling)
            batch_gnn_emb = []
            for seq in seqs:
                act_embs = gnn_emb_full[seq]
                mean_emb = act_embs.mean(dim=0)
                batch_gnn_emb.append(mean_emb)
            batch_gnn_emb = torch.stack(batch_gnn_emb, dim=0)
            # Replace CLS with projected GNN embedding
            gnn_as_cls = llm.project_gnn(batch_gnn_emb).unsqueeze(1)
            fused_embeds = torch.cat([gnn_as_cls, token_embs[:, 1:, :]], dim=1)
            outputs = llm.llm(inputs_embeds=fused_embeds, attention_mask=attention_mask)
            cls_output = outputs.last_hidden_state[:, 0]
            logits = llm.classifier(cls_output)
        else:
            # Pure LLM path: standard forward
            outputs = llm.llm(input_ids=input_ids, attention_mask=attention_mask)
            cls_output = outputs.last_hidden_state[:, 0]
            logits = llm.classifier(cls_output)
        loss = F.cross_entropy(logits, labels)
        preds = logits.argmax(dim=1)
        acc = (preds == labels).float().mean().item()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for group in optimizer.param_groups for p in group['params']], max_norm=1.0
        )
        optimizer.step()
        total_loss += loss.item()
        total_acc += acc
        total_batches += 1

        writer.add_scalar('Loss/iteration', loss.item(), global_step)
        writer.add_scalar('Acc/iteration', acc, global_step)
        global_step += 1

    avg_loss = total_loss / total_batches
    avg_acc = total_acc / total_batches
    writer.add_scalar('Loss/train', avg_loss, epoch)
    writer.add_scalar('Acc/train', avg_acc, epoch)
    print(f"Epoch {epoch+1}/{config['training']['num_epochs']} Loss: {avg_loss:.4f} Acc: {avg_acc:.4f}")

    # Add validation set evaluation
    if use_gnn:
        gnn.eval()
    llm.eval()
    val_loss = 0
    val_acc = 0
    val_batches = 0
    
    # Create validation set DataLoader
    val_loader = DataLoader(datasets['val'], batch_size=config['data']['batch_size'], 
                           shuffle=False, collate_fn=collate_fn)
    
    val_preds = []
    val_true = []
    with torch.no_grad():  # Turn off gradient computation
        for batch_idx, (seqs, labels, timestamps) in enumerate(val_loader):
            seqs, labels = seqs.to(device), labels.to(device)
            
            # Construct text input
            seq_texts = []
            for seq, ts_list in zip(seqs.cpu().tolist(), timestamps):
                acts = [idx_to_act_all.get(i, f"Unknown_Activity_{i}") for i in seq]
                ts_list = [pd.Timestamp(t).to_pydatetime() if not isinstance(t, datetime) else t for t in ts_list]
                sentence = text_processor.sequence_to_sentence_with_time(acts, ts_list, act_to_text)
                seq_texts.append(sentence)

            # Tokenizer encoding into token ids
            inputs = tokenizer(seq_texts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)

            input_ids, attention_mask = inputs["input_ids"].to(device), inputs["attention_mask"].to(device)

            if use_gnn:
                gnn_emb_full = gnn(graph_data)
                with torch.no_grad():
                    token_embs = llm.llm.get_input_embeddings()(input_ids)
                batch_gnn_emb = []
                for seq in seqs:
                    act_embs = gnn_emb_full[seq]
                    mean_emb = act_embs.mean(dim=0)
                    batch_gnn_emb.append(mean_emb)
                batch_gnn_emb = torch.stack(batch_gnn_emb, dim=0)
                gnn_as_cls = llm.project_gnn(batch_gnn_emb).unsqueeze(1)
                fused_embeds = torch.cat([gnn_as_cls, token_embs[:, 1:, :]], dim=1)
                outputs = llm.llm(inputs_embeds=fused_embeds, attention_mask=attention_mask)
                cls_output = outputs.last_hidden_state[:, 0]
                logits = llm.classifier(cls_output)
            else:
                outputs = llm.llm(input_ids=input_ids, attention_mask=attention_mask)
                cls_output = outputs.last_hidden_state[:, 0]
                logits = llm.classifier(cls_output)
            loss = F.cross_entropy(logits, labels)
            preds = logits.argmax(dim=1)
            acc = (preds == labels).float().mean().item()
            
            val_loss += loss.item()
            val_acc += acc
            val_batches += 1

            val_preds.extend(preds.cpu().numpy())
            val_true.extend(labels.cpu().numpy())
    
    # Calculate average values
    avg_val_loss = val_loss / val_batches if val_batches > 0 else 0
    avg_val_acc = val_acc / val_batches if val_batches > 0 else 0
    
    # Log to TensorBoard
    writer.add_scalar('Loss/val', avg_val_loss, epoch)
    writer.add_scalar('Acc/val', avg_val_acc, epoch)
    print(f"Validation Loss: {avg_val_loss:.4f} Acc: {avg_val_acc:.4f}")
    
    # Calculate multiple metrics
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    val_f1 = f1_score(val_true, val_preds, average='weighted', zero_division=0)
    val_precision = precision_score(val_true, val_preds, average='weighted', zero_division=0)
    val_recall = recall_score(val_true, val_preds, average='weighted', zero_division=0)

    # Use F1 score instead of accuracy to select model
    if 'best_val_f1' not in locals():
        best_val_f1 = 0.0
        
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        print(f"Saving best model, validation F1 score: {best_val_f1:.4f}")
        save_payload = {
            'llm_state_dict': llm.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'val_f1': best_val_f1,
            'use_gnn': use_gnn
        }
        if use_gnn:
            save_payload['gnn_state_dict'] = gnn.state_dict()
        torch.save(save_payload, os.path.join(log_dir, "best_model.pt"))

writer.close()

print("Training finished!")
print("\n" + "="*50)
print("Starting Test Set Evaluation...")
print("="*50)

# Test set evaluation
test_set = datasets['test']
test_loader = DataLoader(test_set, batch_size=config['data']['batch_size'], shuffle=False, collate_fn=collate_fn)

# Load the best model for testing
best_model_path = os.path.join(log_dir, "best_model.pt")
if os.path.exists(best_model_path):
    print(f"Loading best model from: {best_model_path}")
    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    if use_gnn and 'gnn_state_dict' in checkpoint and gnn is not None:
        gnn.load_state_dict(checkpoint['gnn_state_dict'], strict=False)
    llm.load_state_dict(checkpoint['llm_state_dict'], strict=False)
    print(f"Best model epoch: {checkpoint.get('epoch', 'Unknown')}")
    print(f"Best validation F1 score: {checkpoint.get('val_f1', 'Unknown'):.4f}")
else:
    print("No best model found, using current model weights")

# Set models to evaluation mode
if use_gnn:
    gnn.eval()
llm.eval()
test_loss = 0
test_acc = 0
test_batches = 0
test_preds = []
test_true = []

with torch.no_grad():
    for batch_idx, (seqs, labels, timestamps) in enumerate(test_loader):
        seqs, labels = seqs.to(device), labels.to(device)

    # Construct text input
        seq_texts = []
        for seq, ts_list in zip(seqs.cpu().tolist(), timestamps):
            acts = [idx_to_act_all.get(i, f"Unknown_Activity_{i}") for i in seq]
            ts_list = [pd.Timestamp(t).to_pydatetime() if not isinstance(t, datetime) else t for t in ts_list]
            sentence = text_processor.sequence_to_sentence_with_time(acts, ts_list, act_to_text)
            seq_texts.append(sentence)

        # Tokenizer encoding into token ids
        inputs = tokenizer(seq_texts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)

        input_ids, attention_mask = inputs["input_ids"].to(device), inputs["attention_mask"].to(device)

        if use_gnn:
            gnn_emb_full = gnn(graph_data)
            token_embs = llm.llm.get_input_embeddings()(input_ids)
            batch_gnn_emb = []
            for seq in seqs:
                act_embs = gnn_emb_full[seq]
                mean_emb = act_embs.mean(dim=0)
                batch_gnn_emb.append(mean_emb)
            batch_gnn_emb = torch.stack(batch_gnn_emb, dim=0)
            gnn_as_cls = llm.project_gnn(batch_gnn_emb).unsqueeze(1)
            fused_embeds = torch.cat([gnn_as_cls, token_embs[:, 1:, :]], dim=1)
            outputs = llm.llm(inputs_embeds=fused_embeds, attention_mask=attention_mask)
            cls_output = outputs.last_hidden_state[:, 0]
            logits = llm.classifier(cls_output)
        else:
            outputs = llm.llm(input_ids=input_ids, attention_mask=attention_mask)
            cls_output = outputs.last_hidden_state[:, 0]
            logits = llm.classifier(cls_output)
        loss = F.cross_entropy(logits, labels)
        preds = logits.argmax(dim=1)
        acc = (preds == labels).float().mean().item()

        test_loss += loss.item()
        test_acc += acc
        test_batches += 1

        test_preds.extend(preds.cpu().numpy())
        test_true.extend(labels.cpu().numpy())

# Calculate test metrics
avg_test_loss = test_loss / test_batches if test_batches > 0 else 0
avg_test_acc = test_acc / test_batches if test_batches > 0 else 0

print(f"\n=== Test Results ===")
print(f"Test Loss: {avg_test_loss:.4f}")
print(f"Test Accuracy: {avg_test_acc:.4f}")

# Calculate comprehensive metrics
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
import numpy as np

test_f1 = f1_score(test_true, test_preds, average='weighted', zero_division=0)
test_precision = precision_score(test_true, test_preds, average='weighted', zero_division=0)
test_recall = recall_score(test_true, test_preds, average='weighted', zero_division=0)

print(f"Test F1: {test_f1:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall: {test_recall:.4f}")

# Performance evaluation
if test_f1 > 0.8:
    print("🏆 Excellent test performance!")
elif test_f1 > 0.7:
    print("✅ Good test performance")
elif test_f1 > 0.6:
    print("⚠️ Acceptable performance, but room for improvement")
else:
    print("❌ Poor performance, needs further optimization")

# Detailed classification report
print(f"\n=== Detailed Classification Report ===")
labels_in_test = np.unique(test_true)
print(f"Number of labels in test set: {len(labels_in_test)} / {len(idx_to_act_all)}")

target_names = [idx_to_act_all[i] for i in labels_in_test]
print(classification_report(test_true, test_preds, labels=labels_in_test, 
                          target_names=target_names, digits=4, zero_division=0))

# Check prediction distribution
unique_pred = np.unique(test_preds)
missing_in_pred = set(labels_in_test) - set(unique_pred)
if missing_in_pred:
    print(f"\n⚠️ Labels never predicted by model: {missing_in_pred}")
    missing_labels = [idx_to_act_all[i] for i in missing_in_pred]
    print(f"Corresponding label names: {missing_labels}")

# Save test results
test_results = {
    'test_loss': avg_test_loss,
    'test_accuracy': avg_test_acc,
    'test_f1': test_f1,
    'test_precision': test_precision,
    'test_recall': test_recall,
    'predictions': test_preds,
    'true_labels': test_true,
    'label_mapping': idx_to_act_all
}

results_path = os.path.join(log_dir, "test_results.pt")
torch.save(test_results, results_path)
print(f"\nTest results saved to: {results_path}")

print("\nEvaluation completed!")
print("="*50)

