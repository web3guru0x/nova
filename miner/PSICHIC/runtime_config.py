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
    BATCH_SIZE = 512