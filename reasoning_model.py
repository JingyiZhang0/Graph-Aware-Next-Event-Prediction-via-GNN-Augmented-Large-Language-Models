# Title: reasoning_model.py
# Author: Jingyi Zhang
# Date: 2025-12-12
# Filename: reasoning_model.py
# Description: DeepSeek (Ollama) reasoning pipeline for next-activity prediction with TF-IDF retrieval.
import subprocess
import json
import re
import argparse
import os
from datetime import datetime as dt
import pandas as pd
from datetime import datetime
from typing import List, Tuple
from myutils.dataset import load_and_split_data, ActivityDataset
from myutils.text_processing import TextProcessor
from myutils.config_loader import load_config
import math
import time
import numpy as np

"""
DeepSeek-based next activity prediction using local Ollama model (deepseek-r1:1.5b).
Assumptions:
- Ollama is installed and `ollama run deepseek-r1:1.5b` works in shell.
- Model only supports plain text prompt -> we craft few-shot style classification prompt.
- We map label activities to integer indices then ask model to answer exactly one label string.

Steps:
1. Load config & dataframe (reusing existing YAML for path & window_size).
2. Split data using existing load_and_split_data (ensures same stratification as training code).
3. For each split (train for building few-shot examples, test for evaluation):
   - Convert each prefix sequence (window_size length) into natural language sentence with time deltas.
4. Build a compact label description list for the model.
5. Build few-shot examples (cap to avoid huge prompt).
6. For every test instance: ask DeepSeek for the next activity (must output one EXACT label token).
7. Parse prediction; compute accuracy.

"""

DEFAULT_MODEL = "deepseek-r1:1.5b"
# DEFAULT_MODEL = "llama3:8b"
# Current Ollama model used for inference (can be overridden by --model or env OLLAMA_MODEL)
CURRENT_MODEL = DEFAULT_MODEL
MAX_FEWSHOTS = 15  # limit to keep prompt short
TIMEOUT = 30  # seconds per model call

def run_ollama(prompt: str) -> str:
    """Run an Ollama model with the given prompt and return raw text output."""
    try:
        # Use --json to capture full stream; gather last response's response field
        result = subprocess.run(
            ["ollama", "run", CURRENT_MODEL],
            input=prompt.encode('utf-8'),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=TIMEOUT
        )
        if result.returncode != 0:
            raise RuntimeError(f"Ollama error: {result.stderr.decode(errors='ignore')}")
        out_text = result.stdout.decode('utf-8', errors='ignore')
        # Some versions stream JSON lines; fallback to plain text.
        try:
            lines = [json.loads(l) for l in out_text.strip().splitlines() if l.strip().startswith('{')]
            if lines:
                # Concatenate response fields
                collected = ''.join([l.get('response', '') for l in lines])
                if collected.strip():
                    return collected.strip()
        except Exception:
            pass
        return out_text.strip()
    except subprocess.TimeoutExpired:
        return "TIMEOUT"


def build_fewshot_examples(train_df, text_processor: TextProcessor, act_to_text_map, window_size: int, label_set):
    examples = []
    count = 0
    for row in train_df.itertuples():
        seq = getattr(row, 'sequence')
        label = getattr(row, 'label')
        ts_list = getattr(row, 'timestamps')
        if label not in label_set:
            continue
        sentence = text_processor.sequence_to_sentence_with_time(seq, ts_list, act_to_text_map)
        examples.append((sentence, label))
        count += 1
        if count >= MAX_FEWSHOTS:
            break
    return examples


def make_prompt(fewshots, candidate_labels, current_sentence, output_mode="label", concise=False):
    if output_mode == 'index':
        label_block = '\n'.join([f"{i}. {lab}" for i, lab in enumerate(candidate_labels)])
        # Add clear separators between examples and remove numeric indexing in the prefix
        shots_text = '\n---\n'.join([
            f"Example:\nHistory: {s}\nNext Activity Index: {candidate_labels.index(l)}" for (s, l) in fewshots
        ])
        header = ("Predict next activity index. Output ONLY the integer." if concise else
                  "You are an expert process mining assistant. Given an event history description, predict the next activity index.\nOnly output the integer index (no label text, no punctuation).")
        return (f"{header}\n\nCandidate Activities (index -> label):\n{label_block}\n\n{shots_text}\n\nHistory: {current_sentence}\nNext Activity Index:")
    else:
        label_block = '\n'.join(f"- {lab}" for lab in candidate_labels)
        # Add clear separators between examples and remove numeric indexing in the prefix
        shots_text = '\n---\n'.join([f"Example:\nHistory: {s}\nNext Activity: {l}" for (s, l) in fewshots])
        header = ("Predict next activity. Output only one label." if concise else
                  "You are a process mining expert specialized in event log analysis. Given the sequence of executed activities (event history), predict the next activity that most likely occurs according to the process behavior. Use your knowledge of workflow semantics and event sequences.\nChoose exactly one label from the candidate list below. Output only that label — do not include punctuation, explanations, or quotation marks. The output must match the label text exactly as written.")
        return (f"{header}\n\nCandidate Activities:\n{label_block}\n\n{shots_text}\n\nHistory: {current_sentence}\nNext Activity:")


def _strip_think_blocks(raw: str) -> str:
    # Remove DeepSeek reasoning blocks <think>...</think>
    return re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()

def extract_label(output: str, candidate_labels, output_mode="label"):
    if output == "TIMEOUT" or not output:
        return None
    cleaned = _strip_think_blocks(output)
    lines = [l.strip() for l in cleaned.split('\n') if l.strip()]
    # Combine also a single joined line for fallback search
    search_space = lines + [' '.join(lines)]
    if output_mode == 'index':
        for line in search_space:
            m = re.match(r'^(\d+)$', line)
            if not m:
                m = re.match(r'^(\d+)[\.: ]', line)
            if not m:
                m = re.search(r'(\d+)', line)  # fallback: first integer anywhere
            if m:
                idx = int(m.group(1))
                if 0 <= idx < len(candidate_labels):
                    return candidate_labels[idx]
        return None
    # label mode parsing (robust normalization on BOTH sides)
    def _normalize(s: str) -> str:
        if s is None:
            return ''
        # strip quotes and leading list markers like "1)", "-", "*", etc.
        s = s.strip().strip('"\'`')
        s = re.sub(r'^[\-\*\d\)\.:\s]+', '', s)
        # replace all non-alnum/underscore/hyphen with spaces, then collapse spaces
        s = re.sub(r'[^A-Za-z0-9_\-]+', ' ', s)
        s = re.sub(r'\s+', ' ', s)
        return s.strip().lower()

    # Precompute normalized candidates
    norm_cands = [(lab, _normalize(lab)) for lab in candidate_labels]

    # exact line match (verbatim)
    for line in search_space:
        if line in candidate_labels:
            return line

    # exact normalized equality
    for line in search_space:
        nline = _normalize(line)
        for lab, nlab in norm_cands:
            if nline == nlab and nlab != '':
                return lab

    #  normalized substring containment — prefer the longest matching candidate
    best = None
    best_len = -1
    for line in search_space:
        nline = _normalize(line)
        if not nline:
            continue
        for lab, nlab in norm_cands:
            if nlab and nlab in nline:
                if len(nlab) > best_len:
                    best = lab
                    best_len = len(nlab)
    if best is not None:
        return best

    # fallback: original loose containment (case-insensitive, unnormalized)
    for line in search_space:
        low = line.lower()
        for lab in candidate_labels:
            if lab.lower() in low:
                return lab
    return None



def main():
    global MAX_FEWSHOTS
    parser = argparse.ArgumentParser(description="DeepSeek next-activity prediction via Ollama")
    parser.add_argument('--model', default=os.environ.get('OLLAMA_MODEL', DEFAULT_MODEL), help='Ollama model name to run (e.g., deepseek-r1:1.5b or a custom GPU variant)')
    parser.add_argument('--config', default='configs/Reasoning_helpdesk.yaml', help='YAML config path')
    parser.add_argument('--device', default=None, help='Override device to control Ollama GPU/CPU (e.g., cpu, cuda, cuda:0)')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of test samples for quick run')
    parser.add_argument('--fewshots', type=int, default=MAX_FEWSHOTS, help='Override number of few-shot examples')
    parser.add_argument('--retrieval', action='store_true', help='Enable TF-IDF similarity-based dynamic few-shot selection')
    parser.add_argument('--retrieval_mode', choices=['none','tfidf'], default='none', help='Similarity backend (tfidf only)')
    parser.add_argument('--retrieval_text', choices=['sentence','acts'], default='sentence', help='Text form for TF-IDF retrieval (full sentence with time vs. compact activity list)')
    parser.add_argument('--fewshot_pool_limit', type=int, default=500, help='Limit training pool size for retrieval to speed up')
    parser.add_argument('--output_mode', choices=['label', 'index'], default='label', help='Use label text or index output to reduce parsing errors')
    parser.add_argument('--votes', type=int, default=1, help='Number of LLM calls (majority vote)')
    parser.add_argument('--train_lr_baseline', action='store_true', help='Train logistic-regression baseline (TF-IDF)')
    parser.add_argument('--skip_llm', action='store_true', help='Skip LLM prediction phase (only run baselines)')
    parser.add_argument('--print_predictions', action='store_true', help='Print each prediction with gold label')
    parser.add_argument('--fallback_strategy', choices=['none','frequency','random'], default='none', help='Fallback when prediction is unknown')
    # Removed: --unique_fewshot_labels / --no_unique_fewshot_labels / --fallback_retrieval_top (simplified hyperparameters, not used in current logic)
    parser.add_argument('--print_fewshots', action='store_true', help='Print few-shot examples (static ones, and first dynamic retrieval set if enabled)')
    # Tuning & logging
    parser.add_argument('--tune', action='store_true', help='Enable hyperparameter tuning on validation split (fewshots / retrieval / votes)')
    parser.add_argument('--tune_fewshots', default='4,8,12', help='Comma list of few-shot counts to try when --tune')
    parser.add_argument('--tune_votes', default='1,3', help='Comma list of vote counts to try when --tune')
    parser.add_argument('--tune_retrieval_modes', default='none,tfidf', help='Comma list of retrieval modes to try when --tune')
    parser.add_argument('--logdir', default='runs', help='TensorBoard log root directory')
    parser.add_argument('--log_tag', default='', help='Optional tag suffix for TensorBoard run name')
    # Speed optimization flags
    parser.add_argument('--fast', action='store_true', help='Apply speed settings (reduce fewshots, index output, concise prompt)')
    parser.add_argument('--concise_prompt', action='store_true', help='Shorter instruction text')
    parser.add_argument('--limit_labels', type=int, default=None, help='Limit candidate label list to top-N frequent for speed')
    parser.add_argument('--profile', action='store_true', help='Profile timing (retrieval, prompt, llm)')
    parser.add_argument('--progress_interval', type=int, default=5, help='Print progress every N samples when not printing predictions')
    parser.add_argument('--timeout', type=int, default=30, help='Per-call Ollama timeout seconds')
    parser.add_argument('--pretty_labels', action='store_true', help='Show cleaned activity names to LLM and in prints, but evaluate on raw labels')
    parser.add_argument('--metric_interval', type=int, default=200, help='Print interim accuracy/F1 every N samples on test (0 to disable)')
    # TensorBoard text logging of prompts/predictions
    parser.add_argument('--tb_log_text', action='store_true', help='Log prompt and prediction details to TensorBoard')
    parser.add_argument('--tb_log_limit', type=int, default=200, help='Max number of samples to log text for per split to avoid large logs')
    args = parser.parse_args()

    # Apply selected model globally
    global CURRENT_MODEL
    CURRENT_MODEL = args.model

    MAX_FEWSHOTS = args.fewshots
    if args.fast:
        prev_fs = MAX_FEWSHOTS
        MAX_FEWSHOTS = min(5, MAX_FEWSHOTS)
        args.output_mode = 'index'
        args.concise_prompt = True
        args.votes = 1
        print(f"[Fast] Applied fast profile: fewshots {prev_fs}->{MAX_FEWSHOTS}, output_mode=index, votes=1, concise_prompt=True")
        if args.timeout == 30:
            args.timeout = 15
    # Apply timeout to global
    global TIMEOUT
    TIMEOUT = max(5, int(args.timeout))
    # Resolve config path (allow passing just filename like BPIC2012.yaml)
    cfg_input = args.config
    cfg_path = cfg_input
    if not os.path.exists(cfg_path):
        # If no directory provided, search in ./configs case-insensitively
        if os.path.dirname(cfg_path) == '':
            cfg_dir = 'configs'
            if os.path.isdir(cfg_dir):
                import glob
                target_lower = os.path.basename(cfg_path).lower()
                matches = [f for f in glob.glob(os.path.join(cfg_dir, '*.yaml')) if os.path.basename(f).lower() == target_lower]
                if matches:
                    cfg_path = matches[0]
                else:
                    tentative = os.path.join(cfg_dir, cfg_path)
                    if os.path.exists(tentative):
                        cfg_path = tentative
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config file not found: {cfg_input} (resolved tried path: {cfg_path}). Use --config configs/XXX.yaml")
    config = load_config(path=cfg_path)
    df = pd.read_csv(config['data']['path'])
    try:
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='ISO8601')
    except ValueError:
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')

    window_size = config['data']['window_size']

    # Determine device from CLI or config and configure Ollama GPU/CPU behavior
    desired_device = getattr(args, 'device', None) if hasattr(args, 'device') else None
    if not desired_device:
        # Try common locations in config
        desired_device = (
            config.get('device')
            or config.get('model', {}).get('device')
            or config.get('training', {}).get('device')
        )
    if desired_device:
        dev = str(desired_device).lower()
        use_gpu = any(k in dev for k in ['cuda', 'gpu', 'mps', 'rocm', 'dml']) and ('cpu' not in dev)
        if use_gpu:
            # Ensure GPU path is enabled
            if 'OLLAMA_NO_GPU' in os.environ:
                os.environ.pop('OLLAMA_NO_GPU', None)
            # Optionally honor specific GPU index like cuda:1
            m = re.search(r':(\d+)', dev)
            if m:
                os.environ['CUDA_VISIBLE_DEVICES'] = m.group(1)
            print(f"[Device] Using GPU for Ollama (device={dev}).", flush=True)
        else:
            os.environ['OLLAMA_NO_GPU'] = '1'
            print("[Device] Forcing CPU for Ollama (device=cpu).", flush=True)

    # Use existing splitter to get sequences
    datasets = load_and_split_data(df, window_size=window_size)
    label_acts = datasets['label_activities']  # all valid labels (raw)

    # Build DataFrames of train/test for easier iteration with raw sequences
    def dataset_to_rows(act_dataset: 'ActivityDataset'):
        rows = []
        for i in range(len(act_dataset)):
            seq_idx, label_idx, ts_seq = act_dataset[i]
            # reconstruct original activity names from seq indices (these indices may include extras). We stored mapping in seq_act_to_idx inside dataset, need reverse:
            rev_map = {v: k for k, v in act_dataset.seq_act_to_idx.items()}
            seq_names = [rev_map[int(x)] for x in seq_idx.tolist()]
            label_name = [k for k, v in datasets['act_to_idx'].items() if v == int(label_idx) ][0]
            rows.append({'sequence': seq_names, 'label': label_name, 'timestamps': ts_seq})
        return pd.DataFrame(rows)

    train_rows = dataset_to_rows(datasets['train'])
    val_rows = dataset_to_rows(datasets['val']) if 'val' in datasets else pd.DataFrame(columns=['sequence','label','timestamps'])
    test_rows = dataset_to_rows(datasets['test'])
    print(f"[Info] Train: {len(train_rows)} | Val: {len(val_rows)} | Test: {len(test_rows)} | Labels: {len(label_acts)}")

    # Build a simple Directly-Follows Graph (DFG) from training prefixes: last_activity -> next_label counts
    from collections import defaultdict
    dfg_edge_counts: dict[tuple[str, str], int] = defaultdict(int)
    next_by_last: dict[str, set[str]] = defaultdict(set)
    nodes_set: set[str] = set()
    for r in train_rows.itertuples():
        seq = getattr(r, 'sequence')
        lab = getattr(r, 'label')
        if not seq:
            continue
        last_a = seq[-1]
        nodes_set.add(last_a); nodes_set.add(lab)
        dfg_edge_counts[(last_a, lab)] += 1
        next_by_last[last_a].add(lab)

    # Undirected adjacency for graph-distance between activities (robust to direction when measuring similarity)
    undirected_adj: dict[str, set[str]] = defaultdict(set)
    for (a, b), _c in dfg_edge_counts.items():
        undirected_adj[a].add(b)
        undirected_adj[b].add(a)

    def _shortest_path_len_undirected(src: str, dst: str) -> int | None:
        if src is None or dst is None:
            return None
        if src == dst:
            return 0
        from collections import deque
        q = deque([(src, 0)])
        seen = {src}
        while q:
            node, d = q.popleft()
            for nb in undirected_adj.get(node, ()): 
                if nb == dst:
                    return d + 1
                if nb not in seen:
                    seen.add(nb)
                    q.append((nb, d + 1))
        return None  # disconnected

    # Text processor and activity rewriting before sentence generation
    data_path_lower = str(config['data']['path']).lower()
    if 'helpdesk' in data_path_lower:
        source = 'helpdesk'
    elif '2012' in data_path_lower or 'bpi12' in data_path_lower:
        source = 'bpic2012'
    elif '2017' in data_path_lower:
        source = 'bpic2017'
    elif '2020' in data_path_lower:
        source = 'bpic2020'
    else:
        source = 'default'
    # Always skip heavy embedding model (pure DeepSeek + TF-IDF pipeline)
    text_processor = TextProcessor(source=source, model_name=config['model']['model_name'], load_model=False)
    # Build mapping for all activities seen across splits + labels
    def _collect_acts(dfrows: pd.DataFrame):
        s = set()
        for r in dfrows.itertuples():
            for a in getattr(r, 'sequence'):
                s.add(a)
        return s
    all_acts = set(label_acts) | _collect_acts(train_rows) | _collect_acts(val_rows) | _collect_acts(test_rows)
    act_to_text = text_processor.clean_activity_names(sorted(all_acts))
    # Quick Ollama preflight (optional)
    if not args.skip_llm and (args.limit is None or args.limit > 0):
        try:
            print(f"[Check] Performing Ollama preflight... (model={CURRENT_MODEL})", flush=True)
            probe = run_ollama("Output only: OK")
            if not probe:
                print("[Warn] Ollama produced no output; model may be downloading or not ready. Try running: ollama run deepseek-r1:1.5b", flush=True)
        except Exception as _e:
            print(f"[Warn] Ollama preflight error: {_e}", flush=True)
    # Build label display mapping
    raw_label_acts = list(label_acts)
    if args.pretty_labels:
        display_labels = [act_to_text.get(x, x) for x in raw_label_acts]
    else:
        display_labels = list(raw_label_acts)
    raw_to_display = {r: d for r, d in zip(raw_label_acts, display_labels)}
    display_to_raw = {d: r for r, d in zip(raw_label_acts, display_labels)}
    # Optional label limiting
    if args.limit_labels and args.limit_labels > 0 and args.limit_labels < len(label_acts):
        from collections import Counter
        # Use training rows to estimate label frequency (train_labels not built yet here)
        train_label_list = train_rows['label'].tolist() if not train_rows.empty else []
        freq = Counter(train_label_list)
        ordered = sorted(label_acts, key=lambda x: (-freq.get(x, 0), x))
        limited = ordered[:args.limit_labels]
        print(f"[Speed] Limiting labels {len(label_acts)} -> {len(limited)} (top-N by freq)")
        label_acts = limited
        # Refresh display lists and mappings to reflect limited label set
        raw_label_acts = list(label_acts)
        if args.pretty_labels:
            display_labels = [act_to_text.get(x, x) for x in raw_label_acts]
        else:
            display_labels = list(raw_label_acts)
        raw_to_display = {r: d for r, d in zip(raw_label_acts, display_labels)}
        display_to_raw = {d: r for r, d in zip(raw_label_acts, display_labels)}

    # Preprocess training text: minimize construction to avoid long waits
    # We split into two types of texts:
    # 1) retrieval_texts used for TF-IDF (can use compact activity list for speed)
    # 2) train_sentences used for few-shot display (generate only when needed)
    if args.retrieval:
        if (args.fewshot_pool_limit is not None and args.fewshot_pool_limit > 0 and len(train_rows) > args.fewshot_pool_limit):
            print(f"[Speed] Retrieval enabled; using first {args.fewshot_pool_limit} training samples as retrieval pool", flush=True)
            retrieval_rows = train_rows.head(args.fewshot_pool_limit)
        else:
            print(f"[Retrieval] Using all {len(train_rows)} training samples as retrieval pool", flush=True)
            retrieval_rows = train_rows
    else:
        retrieval_rows = pd.DataFrame(columns=train_rows.columns)

    # Build retrieval texts (fast path if using 'acts')
    retrieval_texts: List[str] = []
    retrieval_labels: List[str] = []
    if args.retrieval:
        t0 = time.time()
        if args.retrieval_text == 'acts':
            for r in retrieval_rows.itertuples():
                seq = getattr(r, 'sequence')
                lbl = getattr(r, 'label')
                # compact: space-joined cleaned acts (no time)
                cleaned = [act_to_text.get(a, a).lower() for a in seq]
                retrieval_texts.append(' '.join(cleaned))
                retrieval_labels.append(lbl)
        else:
            for i, r in enumerate(retrieval_rows.itertuples(), start=1):
                seq = getattr(r, 'sequence')
                lbl = getattr(r, 'label')
                ts_list = getattr(r, 'timestamps')
                sent = text_processor.sequence_to_sentence_with_time(seq, ts_list, act_to_text)
                retrieval_texts.append(sent)
                retrieval_labels.append(lbl)
                if i % 5000 == 0:
                    print(f"[Build] Constructed {i} training sentences", flush=True)
        print(f"[Build] Built {len(retrieval_texts)} retrieval texts in {time.time()-t0:.1f}s (mode={args.retrieval_text})", flush=True)

    # For static few-shot (no retrieval), build only what we need
    train_sentences: List[str] = []
    train_labels: List[str] = []
    if not args.retrieval:
        need = max(1, min(MAX_FEWSHOTS, len(train_rows)))
        for r in train_rows.head(need).itertuples():
            seq = getattr(r, 'sequence'); ts_list = getattr(r, 'timestamps'); lab = getattr(r, 'label')
            train_sentences.append(text_processor.sequence_to_sentence_with_time(seq, ts_list, act_to_text))
            train_labels.append(lab)

    if not args.retrieval:
        fewshots = list(zip(train_sentences, train_labels))[:MAX_FEWSHOTS]
    else:
        fewshots = []  # dynamic each query

    # Print static few-shot examples if requested (non-retrieval mode)
    if args.print_fewshots and not args.retrieval and fewshots:
        print("\n=== Static Few-Shot Examples ===")
        for i, (sent, lab) in enumerate(fewshots):
            disp_lab = raw_to_display.get(lab, lab)
            preview = (sent[:300] + '...') if len(sent) > 303 else sent
            print(f"[FS {i+1}] Label: {disp_lab}\nHistory: {preview}\n")

    if not fewshots and not args.retrieval and not args.train_lr_baseline:
        print("No static few-shot examples; abort.")
        return

    # Optional logistic regression baseline (embedding classifier)
    # Prepare TF-IDF if needed for retrieval or baseline
    tfidf_vectorizer = None
    tfidf_train_mat = None
    if (args.retrieval and args.retrieval_mode == 'tfidf') or (args.train_lr_baseline and args.retrieval_mode == 'tfidf') or args.tune:
        from sklearn.feature_extraction.text import TfidfVectorizer
        tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
        # fit on retrieval_texts when using retrieval; otherwise on train_sentences (static fs)
        corpus = retrieval_texts if (args.retrieval and len(retrieval_texts) > 0) else train_sentences
        tfidf_train_mat = tfidf_vectorizer.fit_transform(corpus)  # sparse
        # Precompute train vector norms for fast cosine
        import numpy as np
        train_norms = np.sqrt(tfidf_train_mat.power(2).sum(axis=1)).A1
    else:
        train_norms = None

    if args.train_lr_baseline:
        if args.retrieval_mode != 'tfidf':
            print('[Baseline] Set --retrieval_mode tfidf to enable logistic baseline (skipped).')
        else:
            # Fit baseline on the FULL training set (not just few-shots) to avoid single-class issues
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import LabelEncoder
            from sklearn.metrics import accuracy_score
            from sklearn.feature_extraction.text import TfidfVectorizer

            # Build full training corpus
            train_sentences_full: List[str] = []
            train_labels_full: List[str] = []
            for row in train_rows.itertuples():
                seq = getattr(row, 'sequence')
                lab = getattr(row, 'label')
                ts_list = getattr(row, 'timestamps')
                sent = text_processor.sequence_to_sentence_with_time(seq, ts_list, act_to_text)
                train_sentences_full.append(sent)
                train_labels_full.append(lab)

            n_classes = len(set(train_labels_full))
            if n_classes < 2:
                print(f"[Baseline] Training set has only {n_classes} class — skipping LogisticRegression baseline.")
            else:
                vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
                X_train = vec.fit_transform(train_sentences_full)
                le = LabelEncoder()
                y_train = le.fit_transform(train_labels_full)
                clf = LogisticRegression(max_iter=1000)
                clf.fit(X_train, y_train)

                # Build test corpus
                test_sentences = []
                test_labels_str = []
                for row in test_rows.itertuples():
                    seq = getattr(row, 'sequence')
                    lab = getattr(row, 'label')
                    ts_list = getattr(row, 'timestamps')
                    sent = text_processor.sequence_to_sentence_with_time(seq, ts_list, act_to_text)
                    test_sentences.append(sent)
                    test_labels_str.append(lab)
                X_test = vec.transform(test_sentences)
                y_true = le.transform(test_labels_str)
                y_pred = clf.predict(X_test)
                acc_lr = accuracy_score(y_true, y_pred)
                print(f"[Baseline LogisticRegression (TF-IDF) Accuracy] {acc_lr:.4f}")

    # Prototype baseline removed (e5 embeddings no longer used).

    from sklearn.metrics import f1_score, accuracy_score

    # Helper to evaluate a row set with given hyperparams
    def evaluate_split(row_df: pd.DataFrame, fewshot_k: int, retrieval_mode: str, votes: int, output_mode: str,
                       limit: int | None = None, print_preds: bool = False, interim_interval: int | None = None,
                       split_name: str = 'test'):
        nonlocal tfidf_vectorizer, tfidf_train_mat, train_norms
        # Build static fewshots (if retrieval disabled for this eval)
        static_fewshots = list(zip(train_sentences, train_labels))[:fewshot_k]
        preds: List[str] = []
        trues: List[str] = []
        last_acts_eval: List[str] = []
        unknown_local = 0
        sample_times: List[float] = []
        retrieval_times: List[float] = []
        llm_times: List[float] = []
        prompt_times: List[float] = []
        # Per-step accuracy curve
        tb_writer = get_writer()
        correct_so_far = 0
        # Iterate rows
        for i, row in enumerate(row_df.itertuples()):
            if limit is not None and limit > 0 and i >= limit:
                break
            t_sample_start = time.time()
            seq = getattr(row, 'sequence')
            lab = getattr(row, 'label')
            ts_list = getattr(row, 'timestamps')
            sentence = text_processor.sequence_to_sentence_with_time(seq, ts_list, act_to_text)
            last_act_here = seq[-1] if len(seq) > 0 else None
            cur_few = static_fewshots
            if retrieval_mode == 'tfidf':
                if tfidf_vectorizer is None:
                    from sklearn.feature_extraction.text import TfidfVectorizer
                    tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
                    corpus = retrieval_texts if (args.retrieval and len(retrieval_texts) > 0) else train_sentences
                    tfidf_train_mat = tfidf_vectorizer.fit_transform(corpus)
                    train_norms = np.sqrt(tfidf_train_mat.power(2).sum(axis=1)).A1
                t_ret_start = time.time()
                # build retrieval query text in the same style as training
                if args.retrieval_text == 'acts':
                    q_text = ' '.join([act_to_text.get(a, a).lower() for a in seq])
                else:
                    q_text = sentence
                q_vec = tfidf_vectorizer.transform([q_text])
                train_norms_local = train_norms if (train_norms is not None and len(train_norms) == tfidf_train_mat.shape[0]) else np.sqrt(
                    tfidf_train_mat.power(2).sum(axis=1)).A1
                q_norm = math.sqrt(q_vec.power(2).sum()) + 1e-12
                sims = (q_vec @ tfidf_train_mat.T).toarray()[0] / (q_norm * (train_norms_local + 1e-12))
                if fewshot_k < len(sims):
                    top_idx_part = np.argpartition(-sims, fewshot_k)[:fewshot_k]
                    top_idx = top_idx_part[np.argsort(-sims[top_idx_part])]
                else:
                    top_idx = np.argsort(-sims)
                retrieval_times.append(time.time() - t_ret_start)
                # Prioritize candidates whose last activity matches current last activity
                ordered_idx = top_idx
                if last_act_here is not None:
                    same_last = []
                    diff_last = []
                    for idx in top_idx:
                        try:
                            r_seq = retrieval_rows.iloc[int(idx)]['sequence']
                            r_last = r_seq[-1] if (isinstance(r_seq, list) and len(r_seq) > 0) else None
                        except Exception:
                            r_last = None
                        if r_last == last_act_here:
                            same_last.append(idx)
                        else:
                            diff_last.append(idx)
                    ordered_idx = np.array(same_last + diff_last)
                # Cap to fewshot_k
                if fewshot_k < len(ordered_idx):
                    ordered_idx = ordered_idx[:fewshot_k]
                # Build the few-shot sentences on demand to avoid building all upfront
                cur_few = []
                for idx in ordered_idx:
                    if args.retrieval_text == 'acts' and (len(train_sentences) <= idx or len(train_sentences) == 0):
                        # lazily construct the full sentence from original row
                        r = retrieval_rows.iloc[int(idx)]
                        cur_few.append((text_processor.sequence_to_sentence_with_time(r['sequence'], r['timestamps'], act_to_text), r['label']))
                    else:
                        # fallback to existing sentence arrays
                        if len(train_sentences) > idx:
                            cur_few.append((train_sentences[idx], train_labels[idx]))
                        else:
                            r = retrieval_rows.iloc[int(idx)]
                            cur_few.append((text_processor.sequence_to_sentence_with_time(r['sequence'], r['timestamps'], act_to_text), r['label']))
            t_prompt_start = time.time()
            # Convert fewshots labels to display text for the prompt
            cur_few_disp = [(s, raw_to_display.get(l, l)) for (s, l) in cur_few]
            prompt = make_prompt(cur_few_disp, display_labels, sentence, output_mode=output_mode,
                                 concise=(args.concise_prompt or args.fast))
            prompt_times.append(time.time() - t_prompt_start)
            vote_counts = {}
            last_output = None
            t_llm_start = time.time()
            for v in range(votes):
                out = run_ollama(prompt)
                last_output = out
                pred_disp = extract_label(out, display_labels, output_mode=output_mode)
                pred_raw = display_to_raw.get(pred_disp, pred_disp) if pred_disp is not None else None
                if pred_raw is not None:
                    vote_counts[pred_raw] = vote_counts.get(pred_raw, 0) + 1
            llm_times.append(time.time() - t_llm_start)
            if args.profile:
                print(
                    f"[Time] sample#{i + 1}: retrieval={retrieval_times[-1] if retrieval_times else 0:.2f}s prompt={prompt_times[-1]:.2f}s llm={llm_times[-1]:.2f}s",
                    flush=True)
            if votes > 1 and vote_counts:
                pred_label = max(vote_counts.items(), key=lambda x: (x[1], train_labels.count(x[0])))[0]
            else:
                pred_disp_last = extract_label(last_output, display_labels, output_mode=output_mode)
                pred_label = display_to_raw.get(pred_disp_last, pred_disp_last) if pred_disp_last is not None else None
            if pred_label is None:
                if args.fallback_strategy == 'frequency':
                    from collections import Counter
                    freq = Counter(train_labels)
                    pred_label = max(freq.items(), key=lambda x: x[1])[0]
                elif args.fallback_strategy == 'random':
                    import random
                    pred_label = random.choice(label_acts)
                else:
                    unknown_local += 1
            preds.append(pred_label if pred_label is not None else '_UNKNOWN_')
            trues.append(lab)
            last_acts_eval.append(last_act_here)
            # Update and log step-wise accuracy curve
            if pred_label is not None and pred_label == lab:
                correct_so_far += 1
            step_n = i + 1
            if tb_writer:
                try:
                    tb_writer.add_scalar('accuracy', correct_so_far / step_n, step_n)
                except Exception:
                    pass
            if print_preds and args.print_predictions:
                short_hist = (sentence[:120] + '...') if len(sentence) > 123 else sentence
                lab_disp = raw_to_display.get(lab, lab)
                pred_disp_out = raw_to_display.get(pred_label, pred_label) if pred_label is not None else 'UNKNOWN'
                print(f"[Eval] True: {lab_disp} | Pred: {pred_disp_out} | Hist: {short_hist}")
            # TensorBoard text logging (prompt + prediction), limited by args.tb_log_limit
            if args.tb_log_text and i < max(0, int(args.tb_log_limit)):
                w = get_writer()
                if w is not None:
                    # Truncate very long strings to keep logs light
                    def _truncate(s: str, max_len: int = 4000):
                        return s if len(s) <= max_len else (s[:max_len] + '... [truncated]')
                    lab_disp = raw_to_display.get(lab, lab)
                    pred_disp_out = raw_to_display.get(pred_label, pred_label) if pred_label is not None else 'UNKNOWN'
                    cleaned_last = _strip_think_blocks(last_output or '')
                    w.add_text(f"{split_name}/prompt", _truncate(prompt), global_step=i)
                    w.add_text(
                        f"{split_name}/prediction",
                        _truncate(
                            f"True: {lab_disp}\nPred: {pred_disp_out}\nUnknown: {pred_label is None}\nOutput(clean): {cleaned_last}\n"
                        ),
                        global_step=i,
                    )
            sample_times.append(time.time() - t_sample_start)
            if (not args.print_predictions) and args.profile and (i + 1) % max(1, args.progress_interval) == 0:
                avg = sum(sample_times) / len(sample_times)
                print(
                    f"[Progress] {i + 1} samples | avg_sample={avg:.2f}s avg_llm={(sum(llm_times) / len(llm_times)) if llm_times else 0:.2f}s",
                    flush=True)
            # Interim metrics every N samples (if enabled)
            if interim_interval and (i + 1) % interim_interval == 0:
                so_far_true = trues
                so_far_pred = preds
                acc_so_far = accuracy_score(so_far_true, so_far_pred)
                labels_for_f1_sf = sorted(list(set([l for l in so_far_true if l in label_acts])))
                f1_micro_sf = f1_score(so_far_true, so_far_pred, labels=labels_for_f1_sf, average='micro', zero_division=0)
                f1_macro_sf = f1_score(so_far_true, so_far_pred, labels=labels_for_f1_sf, average='macro', zero_division=0)
                print(f"[Interim] {i + 1} samples | Acc={acc_so_far:.4f} F1-micro={f1_micro_sf:.4f} F1-macro={f1_macro_sf:.4f}",
                      flush=True)
        # Compute metrics
        y_true = trues
        y_pred = preds
        acc = accuracy_score(y_true, y_pred)
        labels_for_f1 = sorted(list(set([l for l in y_true if l in label_acts])))
        f1_micro = f1_score(y_true, y_pred, labels=labels_for_f1, average='micro', zero_division=0)
        f1_macro = f1_score(y_true, y_pred, labels=labels_for_f1, average='macro', zero_division=0)
        # Process-aware metrics
        feas_flags = []
        true_cov_flags = []
        dist_vals = []
        for la, y_t, y_p in zip(last_acts_eval, trues, preds):
            if la is None:
                continue
            feas_flags.append(1 if (y_p in next_by_last.get(la, set())) else 0)
            true_cov_flags.append(1 if (y_t in next_by_last.get(la, set())) else 0)
            if y_p is not None and y_p != '_UNKNOWN_' and y_t is not None and y_t != '_UNKNOWN_':
                d = _shortest_path_len_undirected(y_p, y_t)
                if d is not None:
                    dist_vals.append(d)

        feas_rate = (sum(feas_flags) / len(feas_flags)) if feas_flags else 0.0
        true_enabled_rate = (sum(true_cov_flags) / len(true_cov_flags)) if true_cov_flags else 0.0
        avg_graph_distance = (sum(dist_vals) / len(dist_vals)) if dist_vals else None

        return {
            'accuracy': acc,
            'f1_micro': f1_micro,
            'f1_macro': f1_macro,
            'unknown': unknown_local,
            'total': len(y_true),
            'preds': y_pred,
            'trues': y_true,
            'process_metrics': {
                'pred_enabled_rate_last': feas_rate,
                'true_enabled_rate_last': true_enabled_rate,
                'avg_pred_true_graph_distance': avg_graph_distance,
            },
            'profile': {
                'avg_sample_time': (sum(sample_times) / len(sample_times)) if sample_times else 0.0,
                'avg_retrieval_time': (sum(retrieval_times) / len(retrieval_times)) if retrieval_times else 0.0,
                'avg_llm_time': (sum(llm_times) / len(llm_times)) if llm_times else 0.0,
                'avg_prompt_time': (sum(prompt_times) / len(prompt_times)) if prompt_times else 0.0,
                'samples_profiled': len(sample_times)
            } if args.profile else None
        }

    # TensorBoard writer (lazy init) — create a per-run subdirectory under either config['logging']['log_base'] or args.logdir
    writer = None
    def get_writer():
        nonlocal writer
        if writer is None:
            # Always create a unique run subfolder to avoid mixing with previous events
            run_name = f"deepseek_{dt.now().strftime('%Y%m%d_%H%M%S')}{('_'+args.log_tag) if args.log_tag else ''}"
            # Prefer explicit base path from config if provided
            base_path = None
            try:
                base_path = config.get('logging', {}).get('log_base', None)
            except Exception:
                base_path = None
            if base_path:
                log_path = os.path.join(base_path, run_name)
            else:
                log_path = os.path.join(args.logdir, run_name)
            try:
                from torch.utils.tensorboard import SummaryWriter
                os.makedirs(log_path, exist_ok=True)
                writer = SummaryWriter(log_dir=log_path)
                print(f"[TB] Logging to {log_path}")
            except Exception as e:
                print(f"[TB] Could not create SummaryWriter: {e}")
        return writer

    # Hyperparameter tuning on validation split
    best_cfg = None
    if args.tune and len(val_rows) > 0:
        fewshots_list = [int(x) for x in args.tune_fewshots.split(',') if x.strip()]
        votes_list = [int(x) for x in args.tune_votes.split(',') if x.strip()]
        retrieval_modes_list = [m.strip() for m in args.tune_retrieval_modes.split(',') if m.strip()]
        print(f"[Tune] Grid sizes: fewshots={fewshots_list} votes={votes_list} retrieval={retrieval_modes_list}")
        best_score = -1.0
        trial = 0
        for rmode in retrieval_modes_list:
            # Ensure tfidf materials prepared if needed
            if rmode == 'tfidf' and tfidf_vectorizer is None:
                from sklearn.feature_extraction.text import TfidfVectorizer
                tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
                tfidf_train_mat = tfidf_vectorizer.fit_transform(train_sentences)
            for k in fewshots_list:
                for vv in votes_list:
                    trial += 1
                    print(f"[Tune][Trial {trial}] mode={rmode} fewshots={k} votes={vv}")
                    metrics = evaluate_split(val_rows, k, rmode, vv, args.output_mode, limit=None, print_preds=False, split_name='val')
                    print(f"  -> Acc={metrics['accuracy']:.4f} F1_micro={metrics['f1_micro']:.4f} F1_macro={metrics['f1_macro']:.4f} unknown={metrics['unknown']}")
                    # Selection criterion: highest F1_micro then accuracy
                    score = (metrics['f1_micro'], metrics['accuracy'])
                    if score > (best_cfg['score'] if best_cfg else (-1,-1)):
                        best_cfg = {
                            'fewshots': k,
                            'retrieval_mode': rmode,
                            'votes': vv,
                            'metrics': metrics,
                            'score': score
                        }
        if best_cfg:
            print(f"[Tune] Best: mode={best_cfg['retrieval_mode']} fewshots={best_cfg['fewshots']} votes={best_cfg['votes']} Acc={best_cfg['metrics']['accuracy']:.4f} F1_micro={best_cfg['metrics']['f1_micro']:.4f}")
            w = get_writer()
            if w:
                w.add_scalar('val/accuracy', best_cfg['metrics']['accuracy'], 0)
                w.add_scalar('val/f1_micro', best_cfg['metrics']['f1_micro'], 0)
                w.add_scalar('val/f1_macro', best_cfg['metrics']['f1_macro'], 0)
        # Override args with best for test phase
        if best_cfg:
            MAX_FEWSHOTS = best_cfg['fewshots']
            args.retrieval_mode = best_cfg['retrieval_mode']
            args.retrieval = (best_cfg['retrieval_mode'] != 'none')
            args.votes = best_cfg['votes']
            # Rebuild static fewshots if needed
            if not args.retrieval:
                fewshots = list(zip(train_sentences, train_labels))[:MAX_FEWSHOTS]

    # Early exit if skipping LLM or limit explicitly set to 0 (after tuning step)
    if args.skip_llm or (args.limit is not None and args.limit == 0):
        print("[Info] Skipping LLM predictions (either --skip_llm set or --limit 0).")
        return

    # Test evaluation with (possibly tuned) hyperparams
    test_metrics = evaluate_split(
        test_rows,
        MAX_FEWSHOTS,
        args.retrieval_mode if args.retrieval else 'none',
        args.votes,
        args.output_mode,
        limit=args.limit,
        print_preds=True,
        interim_interval=(args.metric_interval if (hasattr(args, 'metric_interval') and args.metric_interval and args.metric_interval > 0) else None),
        split_name='test'
    )
    print("\n=== DeepSeek Test Evaluation ===")
    print(f"Samples: {test_metrics['total']}")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"F1-micro: {test_metrics['f1_micro']:.4f} | F1-macro: {test_metrics['f1_macro']:.4f}")
    print(f"Unknown outputs: {test_metrics['unknown']}")
    pm = test_metrics.get('process_metrics') or {}
    if pm:
        print(f"Pred enabled@last: {pm['pred_enabled_rate_last']:.4f} | True enabled@last: {pm['true_enabled_rate_last']:.4f}")
        print(f"Avg graph distance (pred<->true in DFG): {pm['avg_pred_true_graph_distance'] if pm['avg_pred_true_graph_distance'] is not None else 'N/A'}")
    print(f"Few-shots used: {MAX_FEWSHOTS} | Retrieval: {args.retrieval_mode if args.retrieval else 'none'} | Votes: {args.votes}")
    if args.profile and test_metrics.get('profile'):
        prof = test_metrics['profile']
        print(f"[Profile] avg_sample={prof['avg_sample_time']:.2f}s | avg_retrieval={prof['avg_retrieval_time']:.2f}s | avg_prompt={prof['avg_prompt_time']:.2f}s | avg_llm={prof['avg_llm_time']:.2f}s over {prof['samples_profiled']} samples")
    if args.votes > 1:
        print(f"(Voting enabled: {args.votes})")
    if args.retrieval:
        print("(Dynamic retrieval active)")
    w = get_writer()
    if w:
        w.add_scalar('test/accuracy', test_metrics['accuracy'], 0)
        w.add_scalar('test/f1_micro', test_metrics['f1_micro'], 0)
        w.add_scalar('test/f1_macro', test_metrics['f1_macro'], 0)
        w.add_scalar('test/unknown', test_metrics['unknown'], 0)
        if test_metrics.get('process_metrics'):
            w.add_scalar('test/pred_enabled_rate_last', test_metrics['process_metrics']['pred_enabled_rate_last'], 0)
            w.add_scalar('test/true_enabled_rate_last', test_metrics['process_metrics']['true_enabled_rate_last'], 0)
            if test_metrics['process_metrics']['avg_pred_true_graph_distance'] is not None:
                w.add_scalar('test/avg_graph_distance', test_metrics['process_metrics']['avg_pred_true_graph_distance'], 0)
        w.flush()
        print('[TB] Metrics logged.')
    if writer:
        writer.close()

if __name__ == '__main__':
    main()
