# Title: config_loader.py
# Author: Jingyi Zhang
# Date: 2025-12-12
# Filename: myutils/config_loader.py
# Description: YAML configuration loader with device resolution and CUDA diagnostics.

import yaml
import torch
from typing import Dict, Any

def load_config(path: str = 'config.yaml') -> Dict[str, Any]:
    """Load a YAML configuration and attach a `torch.device`.

    Args:
        path (str): Path to YAML config file.

    Returns:
        dict: Parsed configuration with key `device` set to a `torch.device`.
    """
    with open(path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Debug CUDA availability
    print(f"Is CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Number of available GPUs: {torch.cuda.device_count()}")
        print(f"Current GPU: {torch.cuda.get_device_name(0)}")
    
    # Use the device specified in the config (if CUDA is available)
    config['device'] = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Device set to: {config['device']}")
    
    return config

