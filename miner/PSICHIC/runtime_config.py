import os
import torch
from dotenv import load_dotenv

class RuntimeConfig:
    PSICHIC_PATH = os.path.dirname(os.path.abspath(__file__))
    device = os.environ.get("DEVICE_OVERRIDE")
    DEVICE = ["cpu" if device=="cpu" else "cuda:0"][0]
    MODEL_PATH = os.path.join(PSICHIC_PATH, 'trained_weights', 'PDBv2020_PSICHIC')

    TOTAL_VRAM = torch.cuda.get_device_properties(0).total_memory // 1e9  # GB
    FREE_VRAM = torch.cuda.memory_allocated(0) / 1e9  # GB folosit real, nu doar rezervat

    if FREE_VRAM < 1:  # 🔹 Dacă e prea mic, setează manual
        FREE_VRAM = TOTAL_VRAM * 0.5  # Folosește jumătate din VRAM total ca fallback

    if TOTAL_VRAM > 100:  
        BATCH_SIZE = max(512, min(16384, int(FREE_VRAM * 0.8 * 512)))  
    elif TOTAL_VRAM > 40:
        BATCH_SIZE = max(256, min(8192, int(FREE_VRAM * 0.8 * 512)))
    else:
        BATCH_SIZE = max(128, min(4096, int(FREE_VRAM * 0.8 * 512)))

    print(f"🔹 Using BATCH_SIZE={BATCH_SIZE} for {TOTAL_VRAM}GB VRAM (Free: {FREE_VRAM}GB)")
