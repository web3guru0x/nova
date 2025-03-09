import torch
from torch.amp import autocast  # Actualizat import
import pandas as pd
import os

class PsichicWrapper:
    def __init__(self):
        self.runtime_config = RuntimeConfig()
        self.device = self.runtime_config.DEVICE
        self.screen_df = None  # Inițializează screen_df
        
        with open(os.path.join(self.runtime_config.MODEL_PATH, 'config.json'), 'r') as f:
            self.model_config = json.load(f)
        
    def run_challenge_start(self, protein_seq:str):
        torch.cuda.empty_cache()
        self.load_model()
        self.protein_dict = self.initialize_protein(protein_seq)
        self.protein_seq = [protein_seq]  # Asigură-te că e list
        
    def run_validation(self, smiles_list:list) -> pd.DataFrame:
        self.smiles_dict = self.initialize_smiles(smiles_list)
        torch.cuda.empty_cache()
        
        # Creează screen_df aici
        self.screen_df = pd.DataFrame({
            'Protein': [k for k in self.protein_seq for _ in smiles_list],
            'Ligand': [l for l in smiles_list for _ in self.protein_seq],
        })
        
        self.create_screen_loader(self.protein_dict, self.smiles_dict)
        
        self.model.eval()
        with torch.no_grad(), autocast(device_type='cuda', dtype=torch.float16):
            self.screen_df = virtual_screening(
                self.screen_df, 
                self.model, 
                self.screen_loader,
                os.getcwd(),
                save_interpret=False,
                ligand_dict=self.smiles_dict, 
                device=self.device,
                save_cluster=False
            )
        
        return self.screen_df