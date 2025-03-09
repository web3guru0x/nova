# -*- coding: utf-8 -*-
import os
from dotenv import load_dotenv
import torch

load_dotenv()

class RuntimeConfig:
    PSICHIC_PATH = os.path.dirname(os.path.abspath(__file__))
    # Force GPU usage regardless of environment variable
    DEVICE = "cuda:0"
    MODEL_PATH = os.path.join(PSICHIC_PATH, 'trained_weights', 'PDBv2020_PSICHIC')
    # Significantly increased batch size for H200 SXM with 141GB VRAM
    BATCH_SIZE = 4096
    # Number of parallel streams for GPU processing
    NUM_CUDA_STREAMS = 3
    # Use mixed precision for faster computation
    USE_MIXED_PRECISION = True
    # Memory efficient attention for transformer-based models
    MEMORY_EFFICIENT_ATTENTION = True
    # Set optimal TF32 precision (faster on Ampere+ GPUs while maintaining accuracy)
    ENABLE_TF32 = True
    # Set True to enable memory profiling (set to False in production)
    PROFILE_MEMORY = False
    
    @classmethod
    def apply_gpu_optimizations(cls):
        """Apply global GPU optimizations based on config"""
        # Enable TF32 precision for matrix multiplications (faster and almost as accurate as FP32)
        if cls.ENABLE_TF32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        # Enable cuDNN benchmark for optimal performance
        torch.backends.cudnn.benchmark = True
        
        # Set deterministic to False for better performance
        torch.backends.cudnn.deterministic = False
        
        # Set memory allocation strategy
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
        
        # Pre-allocate memory to avoid fragmentation
        if not cls.PROFILE_MEMORY:
            torch.cuda.empty_cache()
            # Reserve a large chunk of memory to prevent fragmentation
            dummy = torch.zeros(1024, 1024, 1024, device=cls.DEVICE)
            del dummy
            torch.cuda.empty_cache()