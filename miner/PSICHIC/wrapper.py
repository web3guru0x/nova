# -*- coding: utf-8 -*-

import json
import os
import pandas as pd
import torch
import bittensor as bt

from .psichic_utils.dataset import ProteinMoleculeDataset
from .psichic_utils.data_utils import DataLoader, virtual_screening
from .psichic_utils import protein_init, ligand_init
from .models.net import net

from .runtime_config import RuntimeConfig

class PsichicWrapper:
    def __init__(self):
        self.runtime_config = RuntimeConfig()
        self.device = self.runtime_config.DEVICE
        self.results_cache = {}
        
        with open(os.path.join(self.runtime_config.MODEL_PATH, 'config.json'), 'r') as f:
            self.model_config = json.load(f)
            
    def load_model(self):
        degree_dict = torch.load(os.path.join(self.runtime_config.MODEL_PATH,
                                              'degree.pt'), 
                                 weights_only=True
                                 )
        param_dict = os.path.join(self.runtime_config.MODEL_PATH, 'model.pt')
        mol_deg, prot_deg = degree_dict['ligand_deg'], degree_dict['protein_deg']
        
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
                         classification_head=self.model_config['tasks']['classification_task'] ,
                         multiclassification_head=self.model_config['tasks']['mclassification_task'],
                         device=self.device).to(self.device)
        self.model.reset_parameters()    
        self.model.load_state_dict(torch.load(param_dict, 
                                              map_location=self.device, 
                                              weights_only=True
                                              )
                                   )
        
    def initialize_protein(self, protein_seq:str) -> dict:
        self.protein_seq = [protein_seq]
        protein_dict = protein_init(self.protein_seq)
        return protein_dict
    
    def initialize_smiles(self, smiles_list:list) -> dict:
        self.smiles_list = smiles_list
        smiles_dict = ligand_init(smiles_list)
        return smiles_dict
    
    def create_screen_loader(self, protein_dict, smiles_dict):
        bt.logging.success(f"Creating screen_df")
        self.screen_df = pd.DataFrame({'Protein': [k for k in self.protein_seq for _ in self.smiles_list],
                                       'Ligand': [l for l in self.smiles_list for _ in self.protein_seq],
                                       })
        
        bt.logging.success(f"Creating dataset")
        dataset = ProteinMoleculeDataset(self.screen_df, 
                                         smiles_dict, 
                                         protein_dict, 
                                         device=self.device
                                         )
        
        num_workers = 16  # Define it as a variable first
        bt.logging.success(f"Creating DataLoader with {num_workers} workers")
        self.screen_loader = DataLoader(dataset,
                                        batch_size=self.runtime_config.BATCH_SIZE,
                                        shuffle=False,
                                        follow_batch=['mol_x', 'clique_x', 'prot_node_aa'],
                                        num_workers=num_workers,         # Mai mulți workers
                                        prefetch_factor=64,      # Pre-încărcare extinsă
                                        pin_memory=True,
                                        persistent_workers=True
                                        )
        
    def run_challenge_start(self, protein_seq:str):
        torch.cuda.empty_cache()
        self.load_model()
        self.protein_dict = self.initialize_protein(protein_seq)
        
    def run_validation(self, smiles_list):
        # Filter out molecules we've already processed
        uncached_smiles = [s for s in smiles_list if s not in self.result_cache]
        
        if uncached_smiles:
            # Only run inference on new molecules
            self.screen_df = pd.DataFrame({
                'Protein': [k for k in self.protein_seq for _ in uncached_smiles],
                'Ligand': [l for l in uncached_smiles for _ in self.protein_seq],
            })
            
            smiles_dict = self.initialize_smiles(uncached_smiles)
            self.create_screen_loader(self.protein_dict, smiles_dict)
            
            new_results = virtual_screening(
                self.screen_df, 
                self.model, 
                self.screen_loader,
                os.getcwd(),
                save_interpret=False,
                ligand_dict=smiles_dict, 
                device=self.device,
                save_cluster=False
            )
            
            # Update cache with new results
            for _, row in new_results.iterrows():
                self.result_cache[row['Ligand']] = row
        
        # Build result DataFrame from cache
        results = []
        for smiles in smiles_list:
            if smiles in self.result_cache:
                results.append(self.result_cache[smiles])
        
        return pd.DataFrame(results) if results else pd.DataFrame()
        