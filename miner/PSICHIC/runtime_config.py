# -*- coding: utf-8 -*-
import os
from dotenv import load_dotenv

load_dotenv()

class RuntimeConfig:
    PSICHIC_PATH = os.path.dirname(os.path.abspath(__file__))
    device = os.environ.get("DEVICE_OVERRIDE")
    DEVICE = "cuda:0"  # Forțat pe primul GPU
    MODEL_PATH = os.path.join(PSICHIC_PATH, 'trained_weights', 'PDBv2020_PSICHIC')
    BATCH_SIZE = 16384
    