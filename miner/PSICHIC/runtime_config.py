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
    
    # Mărește batch_size pentru H200 cu 141GB VRAM
    BATCH_SIZE = 8192  # Valoarea inițială de 2048 crescută la 8192
    
    # Adăugăm parametri pentru optimizarea performanței
    NUM_WORKERS = 32  # Ajustează în funcție de CPU
    PREFETCH_FACTOR = 4
    USE_MIXED_PRECISION = True
    
    # Optimizări pentru CUDA
    if DEVICE.startswith('cuda'):
        # Activează optimizări CUDA
        torch.backends.cudnn.benchmark = True
        # Setează dimensiunea cache-ului CUDA (în MB)
        CUDA_CACHE_SIZE_MB = 4096  # 4GB cache pentru lookups
        # Setează toleranța la imprecizie pentru mixed precision
        CUDA_AMP_TOLERANCE = 1e-4
        
    # Opțiuni pentru caching
    ENABLE_MODEL_CACHING = True
    MOLECULE_CACHE_SIZE = 10000  # Numărul maxim de molecule în cache
    
    # Parametri pentru rețeaua neurală
    MODEL_HIDDEN_DIM_OVERRIDE = 4096  # Dimensiunea hidden layer redusă pentru performanță