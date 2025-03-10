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
        self.cache = {}  # Cache for molecule scores
        
        with open(os.path.join(self.runtime_config.MODEL_PATH, 'config.json'), 'r') as f:
            self.model_config = json.load(f)
            
        # Configure CUDA for better performance if available
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            # Set optimal threading for your system - adjust based on CPU cores
            torch.set_num_threads(8)  
            
    def load_model(self):
        # Activează fallback pentru torch._dynamo errors
        if hasattr(torch, '_dynamo'):
            torch._dynamo.config.suppress_errors = True
            bt.logging.info("torch._dynamo.config.suppress_errors setat la True pentru a permite fallback la eager mode")
            
        degree_dict = torch.load(os.path.join(self.runtime_config.MODEL_PATH,
                                              'degree.pt'), 
                                 weights_only=True
                                 )
        param_dict = os.path.join(self.runtime_config.MODEL_PATH, 'model.pt')
        mol_deg, prot_deg = degree_dict['ligand_deg'], degree_dict['protein_deg']
        
        # Reduce hidden dimensions for faster inference
        hidden_channels = self.model_config['params']['hidden_channels']
        if hidden_channels > 4096:  # Only reduce if it's large
            bt.logging.info(f"Reducing hidden_channels from {hidden_channels} to 4096 for faster inference")
            hidden_channels = 4096
        
        self.model = net(mol_deg, prot_deg,
                         # MOLECULE
                         mol_in_channels=self.model_config['params']['mol_in_channels'],  
                         prot_in_channels=self.model_config['params']['prot_in_channels'], 
                         prot_evo_channels=self.model_config['params']['prot_evo_channels'],
                         hidden_channels=hidden_channels, 
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
        
        # Convert to half precision to speed up computation and reduce memory usage
        self.model = self.model.half()
        
        # Dezactivăm torch.compile pentru a evita problemele de compatibilitate
        if hasattr(torch, 'compile'):
            bt.logging.info("torch.compile disponibil, dar dezactivat pentru compatibilitate")
                
        # Set model to evaluation mode
        self.model.eval()
        
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
    self.screen_df = pd.DataFrame({
        'Protein': [k for k in self.protein_seq for _ in self.smiles_list],
        'Ligand': [l for l in self.smiles_list for _ in self.protein_seq],
    })
    
    bt.logging.success(f"Creating dataset")
        self.cached_dataset = ProteinMoleculeDataset(self.screen_df, 
                                    smiles_dict, 
                                    protein_dict, 
                                    device=self.device
                                    )
        
        # Calculează batch size optim
        num_molecules = len(self.smiles_list)
        if num_molecules <= 128:
            batch_size = num_molecules
        elif num_molecules <= 1024:
            batch_size = 512
        else:
            batch_size = min(self.runtime_config.BATCH_SIZE, num_molecules)
        
        bt.logging.success(f"Creating DataLoader with {self.runtime_config.NUM_WORKERS} workers and batch_size {batch_size}")
        self.screen_loader = DataLoader(self.cached_dataset,
                                    batch_size=batch_size,
                                    shuffle=False,
                                    follow_batch=['mol_x', 'clique_x', 'prot_node_aa'],
                                    num_workers=self.runtime_config.NUM_WORKERS,
                                    prefetch_factor=self.runtime_config.PREFETCH_FACTOR,
                                    pin_memory=True,
                                    persistent_workers=True
                                    )
        
    def run_challenge_start(self, protein_seq:str):
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        self.load_model()
        self.protein_dict = self.initialize_protein(protein_seq)
        
    def run_validation(self, smiles_list:list) -> pd.DataFrame:
        """Optimized validation with caching for DataLoader and batching"""
        # Initialize smiles dictionary
        self.smiles_dict = self.initialize_smiles(smiles_list)
        
        # DataLoader caching strategy
        if hasattr(self, 'cached_dataset') and hasattr(self, 'cached_smiles_dict') and len(smiles_list) < 5000:
            # Compare if the new smiles list is similar to the cached one
            if set(smiles_list).issubset(set(self.cached_smiles_dict.keys())):
                bt.logging.info("Reutilizare dataset și DataLoader pentru batch similar")
                # Create subset DataFrame
                self.screen_df = pd.DataFrame({
                    'Protein': [k for k in self.protein_seq for _ in smiles_list],
                    'Ligand': [l for l in smiles_list for _ in self.protein_seq],
                })
            else:
                # Re-create dataset and DataLoader
                self.create_screen_loader(self.protein_dict, self.smiles_dict)
                self.cached_smiles_dict = self.smiles_dict
        else:
            # Create new dataset and DataLoader
            self.create_screen_loader(self.protein_dict, self.smiles_dict)
            self.cached_smiles_dict = self.smiles_dict
        
        # Run inference
        amp_enabled = torch.cuda.is_available()
        results_df = self.run_inference_with_optimizations(amp_enabled)
        
        return results_df
    
    def run_inference_with_optimizations(self, amp_enabled=True):
        """
        Runs inference using optimized settings including mixed precision and parallel processing.
        
        Args:
            amp_enabled: Whether to use automatic mixed precision
            
        Returns:
            DataFrame with results
        """
        # Create a copy of the screen DataFrame to store results
        results_df = self.screen_df.copy()
        
        # Initialize result columns if they don't exist
        if 'predicted_binding_affinity' not in results_df.columns:
            results_df['predicted_binding_affinity'] = None
        
        # Prepare for batch processing
        batch_results = []
        
        # Run virtual screening with optimizations
        with torch.no_grad():
            for batch in self.screen_loader:
                # Move batch to device
                batch = batch.to(self.device)
                
                # Run inference with autocast for mixed precision - using corrected syntax
                if amp_enabled:
                    with torch.amp.autocast(device_type='cuda', enabled=True):
                        reg_pred, cls_pred, mcls_pred, sp_loss, o_loss, cl_loss, attention_dict = self.model(
                            # Molecule
                            mol_x=batch.mol_x, mol_x_feat=batch.mol_x_feat, bond_x=batch.mol_edge_attr,
                            atom_edge_index=batch.mol_edge_index, clique_x=batch.clique_x, 
                            clique_edge_index=batch.clique_edge_index, atom2clique_index=batch.atom2clique_index,
                            # Protein
                            residue_x=batch.prot_node_aa, residue_evo_x=batch.prot_node_evo,
                            residue_edge_index=batch.prot_edge_index,
                            residue_edge_weight=batch.prot_edge_weight,
                            # Mol-Protein Interaction batch
                            mol_batch=batch.mol_x_batch, prot_batch=batch.prot_node_aa_batch, clique_batch=batch.clique_x_batch,
                            # save_cluster
                            save_cluster=False
                        )
                else:
                    reg_pred, cls_pred, mcls_pred, sp_loss, o_loss, cl_loss, attention_dict = self.model(
                        # Molecule
                        mol_x=batch.mol_x, mol_x_feat=batch.mol_x_feat, bond_x=batch.mol_edge_attr,
                        atom_edge_index=batch.mol_edge_index, clique_x=batch.clique_x, 
                        clique_edge_index=batch.clique_edge_index, atom2clique_index=batch.atom2clique_index,
                        # Protein
                        residue_x=batch.prot_node_aa, residue_evo_x=batch.prot_node_evo,
                        residue_edge_index=batch.prot_edge_index,
                        residue_edge_weight=batch.prot_edge_weight,
                        # Mol-Protein Interaction batch
                        mol_batch=batch.mol_x_batch, prot_batch=batch.prot_node_aa_batch, clique_batch=batch.clique_x_batch,
                        # save_cluster
                        save_cluster=False
                    )
                
                # Extract and process predictions
                batch_size = len(batch.mol_key)
                batch_result = pd.DataFrame({
                    'Protein': batch.prot_key,
                    'Ligand': batch.mol_key
                })
                
                # Process regression predictions if available
                if reg_pred is not None:
                    reg_values = reg_pred.squeeze().reshape(-1).cpu().numpy()
                    batch_result['predicted_binding_affinity'] = reg_values
                
                # Process classification predictions if available
                if cls_pred is not None:
                    cls_values = torch.sigmoid(cls_pred).squeeze().reshape(-1).cpu().numpy()
                    batch_result['predicted_binary_interaction'] = cls_values
                
                # Process multiclass predictions if available
                if mcls_pred is not None:
                    mcls_values = torch.softmax(mcls_pred, dim=-1).cpu().numpy()
                    # Extract class probabilities
                    for i in range(mcls_values.shape[1]):
                        batch_result[f'predicted_class_{i}'] = mcls_values[:, i]
                
                # Add batch results to the list
                batch_results.append(batch_result)
                
                # Clear unnecessary tensors to free memory
                del batch, reg_pred, cls_pred, mcls_pred, attention_dict
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Combine all batch results
        if batch_results:
            results_df = pd.concat(batch_results, ignore_index=True)
        
        return results_df