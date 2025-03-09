# -*- coding: utf-8 -*-

import json
import os
import pandas as pd
import torch

from .psichic_utils.dataset import ProteinMoleculeDataset
from .psichic_utils.data_utils import DataLoader, virtual_screening
from .psichic_utils import protein_init, ligand_init
from .models.net import net

from .runtime_config import RuntimeConfig

class PsichicWrapper:
    def __init__(self):
        self.runtime_config = RuntimeConfig()
        # Forțează utilizarea CUDA dacă este disponibil
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.runtime_config.DEVICE = self.device
        
        # Afișează informații despre CUDA pentru diagnostic
        if torch.cuda.is_available():
            print(f"CUDA available: {torch.cuda.is_available()}")
            print(f"CUDA device count: {torch.cuda.device_count()}")
            print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
            print(f"Using device: {self.device}")
        else:
            print("CUDA not available, using CPU")
        
        with open(os.path.join(self.runtime_config.MODEL_PATH, 'config.json'), 'r') as f:
            self.model_config = json.load(f)
            
    def load_model(self):
        degree_dict = torch.load(os.path.join(self.runtime_config.MODEL_PATH,
                                            'degree.pt'), 
                                weights_only=True,
                                map_location=self.device
                                )
        param_dict = os.path.join(self.runtime_config.MODEL_PATH, 'model.pt')
        mol_deg, prot_deg = degree_dict['ligand_deg'], degree_dict['protein_deg']
        
        # Asigură-te că gradul moleculei și al proteinei sunt pe device-ul corect
        mol_deg = mol_deg.to(self.device)
        prot_deg = prot_deg.to(self.device)
        
        self.model = net(mol_deg, prot_deg,
                        # MOLECULE
                        mol_in_channels=self.model_config['params']['mol_in_channels'],  
                        prot_in_channels=self.model_config['params']['prot_in_channels'], 
                        prot_evo_channels=self.model_config['params']['prot_evo_channels'],
                        hidden_channels=self.model_config['params']['hidden_channels'], 
                        pre_layers=self.model_config['params']['pre_layers'], 
                        post_layers=self.model_config['params']['post_layers'],
                        aggregators=self.model_config['params']['aggregators'], 
                        scalers=self.model_config['params']['scalers'],
                        total_layer=self.model_config['params']['total_layer'],                
                        K=self.model_config['params']['K'],
                        heads=self.model_config['params']['heads'], 
                        dropout=self.model_config['params']['dropout'],
                        dropout_attn_score=self.model_config['params']['dropout_attn_score'],
                        # output
                        regression_head=self.model_config['tasks']['regression_task'],
                        classification_head=self.model_config['tasks']['classification_task'],
                        multiclassification_head=self.model_config['tasks']['mclassification_task'],
                        device=self.device).to(self.device)
        self.model.reset_parameters()
        
        # Încarcă parametrii modelului
        self.model.load_state_dict(torch.load(param_dict, 
                                            map_location=self.device, 
                                            weights_only=True
                                            ))
        
        # Verifică și afișează device-ul modelului pentru confirmare
        print(f"Model loaded on device: {next(self.model.parameters()).device}")
        # Activează cudnn benchmark pentru accelerare
        torch.backends.cudnn.benchmark = True
        
    def initialize_protein(self, protein_seq:str) -> dict:
        self.protein_seq = [protein_seq]
        protein_dict = protein_init(self.protein_seq)
        return protein_dict
    
    def initialize_smiles(self, smiles_list:list) -> dict:
        self.smiles_list = smiles_list
        smiles_dict = ligand_init(smiles_list)
        return smiles_dict
    
    def create_screen_loader(self, protein_dict, smiles_dict):
        self.screen_df = pd.DataFrame({'Protein': [k for k in self.protein_seq for _ in self.smiles_list],
                                       'Ligand': [l for l in self.smiles_list for _ in self.protein_seq],
                                       })
        
        dataset = ProteinMoleculeDataset(self.screen_df, 
                                         smiles_dict, 
                                         protein_dict, 
                                         device=self.device
                                         )
        
        self.screen_loader = DataLoader(dataset,
                                        batch_size=self.runtime_config.BATCH_SIZE,
                                        shuffle=False,
                                        follow_batch=['mol_x', 'clique_x', 'prot_node_aa']
                                        )
        
    def run_challenge_start(self, protein_seq:str):
        torch.cuda.empty_cache()
        self.load_model()
        
        # Verifică dacă secvența proteică este validă
        if protein_seq is None:
            raise ValueError("Protein sequence cannot be None")
        
        print(f"Protein sequence length: {len(protein_seq)}")
        try:
            self.protein_dict = self.initialize_protein(protein_seq)
            if not self.protein_dict:
                raise ValueError(f"Failed to initialize protein dictionary for {protein_seq}")
            print(f"Successfully initialized protein_dict with {len(self.protein_dict)} entries")
        except Exception as e:
            print(f"Error in initialize_protein: {e}")
            raise
        
    def run_validation(self, smiles_list:list) -> pd.DataFrame:
        self.smiles_dict = self.initialize_smiles(smiles_list)
        
        # Clear CUDA cache before starting inference
        if self.device != "cpu" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Create a dataloader that properly uses CUDA
        self.create_screen_loader(self.protein_dict, self.smiles_dict)
        
        # Enable mixed precision with float16 for better GPU performance
        use_amp = True if self.device != "cpu" and torch.cuda.is_available() else False
        amp_dtype = torch.float16 if use_amp else None
        
        # Use CUDA streams and profiling for GPU optimization
        if self.device != "cpu" and torch.cuda.is_available():
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream):
                with torch.amp.autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
                    self.screen_df = virtual_screening(
                        self.screen_df, 
                        self.model, 
                        self.screen_loader,
                        os.getcwd(),
                        save_interpret=False,
                        ligand_dict=self.smiles_dict, 
                        device=self.device,
                        save_cluster=False,
                    )
                # Ensure GPU operations are complete
                torch.cuda.synchronize()
        else:
            self.screen_df = virtual_screening(
                self.screen_df, 
                self.model, 
                self.screen_loader,
                os.getcwd(),
                save_interpret=False,
                ligand_dict=self.smiles_dict, 
                device=self.device,
                save_cluster=False,
            )
        
        return self.screen_df
        
