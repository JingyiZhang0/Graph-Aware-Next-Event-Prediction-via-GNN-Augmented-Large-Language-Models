# Graph-Aware Next Event Prediction via GNN-Augmented Large Language Models

This repository provides three complementary pipelines for next activity (next-event) prediction in business process mining, combining Large Language Models (LLMs) and Graph Neural Networks (GNNs).

## Pipelines

- Hybrid + Pure LLM (`train_predict.py`)
  - Hybrid model: fuses a process-aware GNN encoder with an LLM-based text predictor.
  - Pure LLM: uses an instruction-following prompt to predict the next event directly.
- Pure GNN (`train_gat.py`)
  - Time-aware Graph Attention Network (GAT) over an activity graph, followed by an MLP classifier on pooled sequence embeddings.
- LLM Reasoning Mode (`reasoning_model.py`)
  - Uses a local LLM (Ollama) with robust prompt engineering and reasoning outputs to predict the next activity. Includes TF‑IDF retrieval for dynamic few-shots.

## Project Structure

```text
├── train_predict.py        # Hybrid and pure LLM pipelines
├── train_gat.py            # Pure GNN pipeline (TimeAwareGAT + MLP)
├── reasoning_model.py      # LLM reasoning mode with retrieval and voting
├── models/
│   ├── fusion_model.py     # Hybrid fusion of GNN + LLM representations
│   ├── gnn_encoder.py      # Time-aware GAT encoder
│   └── llm_predictor.py    # LLM predictor utilities
├── myutils/
│   ├── build_graph.py      # Build activity graph with temporal edges
│   ├── dataset.py          # Sliding-window datasets and collate
│   ├── text_processing.py  # Activity text cleaning and sentence generation
│   └── config_loader.py    # YAML config loader
├── configs/                # Dataset-specific YAMLs
├── requirements.txt        # Python dependencies
└── notebooks/              # Data exploration
```

## Setup

- Python 3.8+
- Install dependencies:
  - `pip install -r requirements.txt`
- Optional for reasoning mode:
  - Install [Ollama] and pull a local model (e.g., `ollama run deepseek-r1:1.5b`).

## Configuration

Edit a YAML in `configs/` to set:

- `data.path`: CSV path with columns including `timestamp` and activity IDs/names.
- `data.window_size`: sliding window length.
- Model/training options (learning rates, pooling, device, etc.).

Sample configs:

- `configs/BPIC2012.yaml`, `configs/BPIC2020.yaml`, `configs/helpdesk.yaml`
- Reasoning variants: `configs/Reasoning_*.yaml`

## Quick Start

- Hybrid or Pure LLM (in `train_predict.py`):
  - Run directly with a config:

    ```bash
    python train_predict.py --config configs/BPIC2012.yaml
    ```

  - Select mode via script args/config (see in-file help).

- Pure GNN (in `train_gat.py`):
  - Run directly with a config:

    ```bash
    python train_gat.py --config configs/BPIC2012.yaml
    ```

  - Produces train/val/test metrics (Accuracy, F1, Precision, Recall).

- LLM Reasoning Mode (in `reasoning_model.py`):

    ```bash
    python reasoning_model.py --model deepseek-r1:1.5b --config configs/Reasoning_helpdesk.yaml --output_mode index --fewshots 8 --retrieval --retrieval_mode tfidf --retrieval_text acts --fewshot_pool_limit 500 --votes 3 --metric_interval 200
    ```

  Parameters can be freely chosen and combined; omitting `--limit` will predict the entire test set. `--fast` applies a speed profile (fewer few-shots, index output, concise prompt, single vote, shorter timeout).
