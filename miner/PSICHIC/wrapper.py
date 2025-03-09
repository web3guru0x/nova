# -*- coding: utf-8 -*-

import json
import os
import pandas as pd
import torch
from tqdm import tqdm

from .psichic_utils.dataset import ProteinMoleculeDataset
from .psichic_utils.data_utils import DataLoader, virtual_screening
from .psichic_utils import protein_init, ligand_init
from .models.net import net

from .runtime_config import RuntimeConfig

class PsichicWrapper:
    def __init__(self):
        self.runtime_config = RuntimeConfig()
        self.device = self.runtime_config.DEVICE
        
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
                                        follow_batch=['mol_x', 'clique_x', 'prot_node_aa'],
                                        num_workers=64,         # Increased from 32
                                        prefetch_factor=32,      # Increased from 16
                                        pin_memory=True,
                                        persistent_workers=True
                                        )
        
    def run_challenge_start(self, protein_seq:str):
        torch.cuda.empty_cache()
        self.load_model()
        self.protein_dict = self.initialize_protein(protein_seq)
        
    def run_validation(self, smiles_list:list) -> pd.DataFrame:
        self.smiles_dict = self.initialize_smiles(smiles_list)
        torch.cuda.empty_cache()
        self.create_screen_loader(self.protein_dict, self.smiles_dict)
        
        # Create FP16 scaler for mixed precision
        scaler = torch.cuda.amp.GradScaler()
        
        # Run with mixed precision
        with torch.no_grad():
            reg_preds = []
            cls_preds = []
            mcls_preds = []
            interaction_keys_all = []
            attention_dicts = []
            
            self.model.eval()
            
            for data in tqdm(self.screen_loader):
                data = data.to(self.device)
                
                # Use mixed precision for faster computation
                with torch.cuda.amp.autocast():
                    reg_pred, cls_pred, mcls_pred, sp_loss, o_loss, cl_loss, attention_dict = self.model(
                            # Molecule
                            mol_x=data.mol_x, mol_x_feat=data.mol_x_feat, bond_x=data.mol_edge_attr,
                            atom_edge_index=data.mol_edge_index, clique_x=data.clique_x, 
                            clique_edge_index=data.clique_edge_index, atom2clique_index=data.atom2clique_index,
                            # Protein
                            residue_x=data.prot_node_aa, residue_evo_x=data.prot_node_evo,
                            residue_edge_index=data.prot_edge_index,
                            residue_edge_weight=data.prot_edge_weight,
                            # Mol-Protein Interaction batch
                            mol_batch=data.mol_x_batch, prot_batch=data.prot_node_aa_batch, clique_batch=data.clique_x_batch,
                            # save_cluster
                            save_cluster=False
                    )
                
                interaction_keys = list(zip(data.prot_key, data.mol_key))
                interaction_keys_all.extend(interaction_keys)
                
                if reg_pred is not None:
                    reg_pred = reg_pred.squeeze().reshape(-1).cpu().numpy()
                    reg_preds.append(reg_pred)
                    
                if cls_pred is not None:
                    cls_pred = torch.sigmoid(cls_pred).squeeze().reshape(-1).cpu().numpy()
                    cls_preds.append(cls_pred)

                if mcls_pred is not None:
                    mcls_pred = torch.softmax(mcls_pred,dim=-1).cpu().numpy()
                    mcls_preds.append(mcls_pred)
                    
                attention_dicts.append(attention_dict)
                
            # Combine results
            df_results = pd.DataFrame(interaction_keys_all, columns=['Protein', 'Ligand'])
            if reg_preds:
                all_reg_preds = np.concatenate(reg_preds)
                df_results['predicted_binding_affinity'] = all_reg_preds
            
            if cls_preds:
                all_cls_preds = np.concatenate(cls_preds)
                df_results['predicted_binary_interaction'] = all_cls_preds
                
            # Free memory
            torch.cuda.empty_cache()
            
        return df_results
        
