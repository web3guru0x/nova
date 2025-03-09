# -*- coding: utf-8 -*-
import os
from dotenv import load_dotenv
import torch

load_dotenv()

class RuntimeConfig:
    PSICHIC_PATH = os.path.dirname(os.path.abspath(__file__))
    device = os.environ.get("DEVICE_OVERRIDE")
    # Forțează CUDA dacă este disponibil, ignoră DEVICE_OVERRIDE
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    MODEL_PATH = os.path.join(PSICHIC_PATH, 'trained_weights', 'PDBv2020_PSICHIC')
    BATCH_SIZE = 128