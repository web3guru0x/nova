# -*- coding: utf-8 -*-
import os
import torch
from dotenv import load_dotenv

load_dotenv()

class RuntimeConfig:
    PSICHIC_PATH = os.path.dirname(os.path.abspath(__file__))
    device = os.environ.get("DEVICE_OVERRIDE")
    
    # Detectare automată a GPU sau forțare pe device-ul specific
    if device:
        DEVICE = device
    else:
        DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    MODEL_PATH = os.path.join(PSICHIC_PATH, 'trained_weights', 'PDBv2020_PSICHIC')
    
    # Optimizări extreme pentru batch size pe H200
    BATCH_SIZE = 65536  # Mărit dramatic pentru H200 cu 141GB VRAM
    
    # Optimizări CPU și paralelizare
    NUM_WORKERS = 8  # Redus pentru a evita overhead-ul
    PREFETCH_FACTOR = 1
    USE_MIXED_PRECISION = True
    
    # Optimizări pentru CUDA
    if DEVICE.startswith('cuda'):
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.enabled = True
        torch.set_float32_matmul_precision('high') 
        
    # Model size
    MODEL_HIDDEN_DIM_OVERRIDE = 2048  # Redus semnificativ pentru viteză
    
    # Optimizare paralelă
    PARALLEL_BATCH_PROCESSING = True
    MAX_PARALLEL_WORKERS = 4
    
    # Cache agresiv
    MOLECULE_CACHE_SIZE = 50000  # Mărire cache
    SIMILARITY_THRESHOLD = 0.85  # Threshold redus pentru mai multe cache hits