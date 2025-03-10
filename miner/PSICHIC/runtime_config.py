# -*- coding: utf-8 -*-
import os
from dotenv import load_dotenv

load_dotenv()

class RuntimeConfig:
    PSICHIC_PATH = os.path.dirname(os.path.abspath(__file__))
    device = os.environ.get("DEVICE_OVERRIDE")
    DEVICE = ["cpu" if device=="cpu" else "cuda:0"][0]
    MODEL_PATH = os.path.join(PSICHIC_PATH, 'trained_weights', 'PDBv2020_PSICHIC')
    BATCH_SIZE = 512  # Mărit substanțial pentru H200 cu 140GB memorie
    
    # Optimizări pentru H200
    TORCH_CUDA_ARCH_LIST = "9.0"
    USE_MIXED_PRECISION = True
    NUM_WORKERS = 16  # Mărit numărul de workeri
    PREFETCH_FACTOR = 4  # Prefetch mai multe batch-uri