# -*- coding: utf-8 -*-
import os
from dotenv import load_dotenv
import torch

load_dotenv()

class RuntimeConfig:
    PSICHIC_PATH = os.path.dirname(os.path.abspath(__file__))
    device = os.environ.get("DEVICE_OVERRIDE")
    DEVICE = "cuda:0"  # Forțat pe primul GPU
    MODEL_PATH = os.path.join(PSICHIC_PATH, 'trained_weights', 'PDBv2020_PSICHIC')
    BATCH_SIZE = 16384  # Increased from 4096
    MODEL_CACHE = True  # Added for memory caching
    
    # Enable CUDA optimizations
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True