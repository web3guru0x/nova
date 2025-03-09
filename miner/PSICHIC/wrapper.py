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
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            cuda_device_count = torch.cuda.device_count()
            cuda_device_name = torch.cuda.get_device_name(0)
            print(f"CUDA available: {cuda_available}, Device count: {cuda_device_count}, Device name: {cuda_device_name}")
            # Print memory information
            print(f"GPU total memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB")
            # Set GPU optimization flags
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            # Set CUDA device to 0
            torch.cuda.set_device(0)
        else:
            print("CUDA not available, using CPU")
            
        self.device = "cuda:0" if cuda_available else "cpu"
        self.runtime_config.DEVICE = self.device
        
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
        
        # Optimize the DataLoader for GPU throughput
        self.screen_loader = DataLoader(dataset,
                                        batch_size=min(4096, len(dataset)),  # Smaller batch size to avoid OOM
                                        shuffle=False,
                                        follow_batch=['mol_x', 'clique_x', 'prot_node_aa'],
                                        pin_memory=True,
                                        num_workers=4,  # Increase worker count
                                        prefetch_factor=4,
                                        persistent_workers=True  # Keep workers alive between batches
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
        # Clean GPU memory
        if self.device != "cpu" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Create dictionary on GPU directly
        self.smiles_dict = self.initialize_smiles(smiles_list)
        
        # Create optimized dataloader
        self.create_screen_loader(self.protein_dict, self.smiles_dict)
        
        # Configure mixed precision for better GPU utilization
        scaler = torch.cuda.amp.GradScaler() if self.device != "cpu" else None
        
        # Run inference through a dedicated CUDA stream for better GPU utilization
        if self.device != "cpu" and torch.cuda.is_available():
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream):
                with torch.amp.autocast(device_type='cuda', enabled=True):
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
        
