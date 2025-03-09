# -*- coding: utf-8 -*-

import json
import os
import pandas as pd
import torch
import torch.cuda

from .psichic_utils.dataset import ProteinMoleculeDataset
from .psichic_utils.data_utils import DataLoader, virtual_screening
from .psichic_utils import protein_init, ligand_init
from .models.net import net

from .runtime_config import RuntimeConfig

class PsichicWrapper:
    def __init__(self):
        self.runtime_config = RuntimeConfig()
        self.device = self.runtime_config.DEVICE
        # Apply GPU optimizations
        RuntimeConfig.apply_gpu_optimizations()
        
        # Create CUDA streams for parallel processing
        self.streams = [torch.cuda.Stream() for _ in range(self.runtime_config.NUM_CUDA_STREAMS)]
        self.current_stream_idx = 0
        
        # Create scaler for mixed precision
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.runtime_config.USE_MIXED_PRECISION)
        
        # Pre-allocate CUDA events for timing
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        
        # Load model configuration
        with open(os.path.join(self.runtime_config.MODEL_PATH, 'config.json'), 'r') as f:
            self.model_config = json.load(f)
        
        # Cache for protein embeddings to avoid recomputation
        self.protein_embedding_cache = {}
        
        # Set optimal thread settings
        torch.set_num_threads(8)  # Limit CPU threads to avoid overhead
        self.memory_tracking = True

    def _log_time(self, stage, start_time=None):
        """Helper to log timing information within the wrapper"""
        current_time = time.time()
        if start_time:
            elapsed = current_time - start_time
            print(f"WRAPPER TIMING: {stage} took {elapsed:.4f} seconds")
            return current_time
        return current_time
            
    def get_next_stream(self):
        """Get the next CUDA stream in round-robin fashion"""
        stream = self.streams[self.current_stream_idx]
        self.current_stream_idx = (self.current_stream_idx + 1) % len(self.streams)
        return stream
        
    def load_model(self):
        degree_dict = torch.load(os.path.join(self.runtime_config.MODEL_PATH,
                                              'degree.pt'), 
                                 weights_only=True,
                                 map_location=self.device
                                 )
        param_dict = os.path.join(self.runtime_config.MODEL_PATH, 'model.pt')
        mol_deg, prot_deg = degree_dict['ligand_deg'].cuda(), degree_dict['protein_deg'].cuda()
        
        # Use JIT compilation for critical model components
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
        
        # Reset parameters
        self.model.reset_parameters()
        
        # Load pre-trained weights
        self.model.load_state_dict(torch.load(param_dict, 
                                              map_location=self.device, 
                                              weights_only=True
                                              ))
        
        # Optimize model for inference
        self.model.eval()
        
        # Apply fusion optimization if available in PyTorch version
        if hasattr(torch, 'jit'):
            try:
                self.model = torch.jit.optimize_for_inference(
                    torch.jit.script(self.model)
                )
            except Exception as e:
                print(f"JIT optimization failed: {e}")
        
    def initialize_protein(self, protein_seq:str) -> dict:
        # Check cache first
        if protein_seq in self.protein_embedding_cache:
            return self.protein_embedding_cache[protein_seq]
        
        self.protein_seq = [protein_seq]
        
        # Process protein initialization in a dedicated stream
        with torch.cuda.stream(self.get_next_stream()):
            protein_dict = protein_init(self.protein_seq)
            
            # Cache for future use
            self.protein_embedding_cache = protein_dict
        
        # Wait for stream to complete
        torch.cuda.synchronize()
        
        return protein_dict
    
    def initialize_smiles(self, smiles_list:list) -> dict:
        self.smiles_list = smiles_list
        
        # Process in parallel with a dedicated stream
        stream = self.get_next_stream()
        with torch.cuda.stream(stream):
            # Process in larger batches for better GPU utilization
            batch_size = 256
            smiles_dict = {}
            
            for i in range(0, len(smiles_list), batch_size):
                batch = smiles_list[i:min(i+batch_size, len(smiles_list))]
                batch_dict = ligand_init(batch)
                smiles_dict.update(batch_dict)
            
            # Move tensors to GPU with non-blocking transfer
            for key, value in smiles_dict.items():
                for tensor_key in value:
                    if isinstance(value[tensor_key], torch.Tensor):
                        value[tensor_key] = value[tensor_key].cuda(non_blocking=True)
        
        # Ensure processing is complete
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        
        return smiles_dict
    
    def create_screen_loader(self, protein_dict, smiles_dict):
        """Create optimized data loader with detailed profiling"""
        start_time = time.time()
        print(f"DATALOADER: Creating screen loader with {len(self.smiles_list)} molecules")
        
        # Time DataFrame creation
        df_start = time.time()
        self.screen_df = pd.DataFrame({'Protein': [k for k in self.protein_seq for _ in self.smiles_list],
                                    'Ligand': [l for l in self.smiles_list for _ in self.protein_seq],
                                    })
        df_end = time.time()
        print(f"DATALOADER: DataFrame creation took {df_end - df_start:.4f} seconds")
        
        # Time dataset creation
        dataset_start = time.time()
        dataset = ProteinMoleculeDataset(self.screen_df, 
                                        smiles_dict, 
                                        protein_dict, 
                                        device=self.device
                                        )
        dataset_end = time.time()
        print(f"DATALOADER: Dataset creation took {dataset_end - dataset_start:.4f} seconds")
        
        # Use optimized data loader with pinned memory and multiple workers
        loader_start = time.time()
        
        # Track memory before loader creation
        if hasattr(torch.cuda, 'memory_allocated'):
            mem_before = torch.cuda.memory_allocated() / (1024**3)  # GB
        
        self.screen_loader = DataLoader(dataset,
                                        batch_size=self.runtime_config.BATCH_SIZE,
                                        shuffle=False,
                                        pin_memory=True,
                                        num_workers=8,  # Increased from 4 to 8
                                        persistent_workers=True,
                                        prefetch_factor=5,  # Increased prefetch factor
                                        follow_batch=['mol_x', 'clique_x', 'prot_node_aa']
                                    )
        
        # Track memory after loader creation
        if hasattr(torch.cuda, 'memory_allocated'):
            mem_after = torch.cuda.memory_allocated() / (1024**3)  # GB
            print(f"DATALOADER: Memory usage change: {mem_after - mem_before:.2f}GB")
        
        # Try warming up the dataloader to prevent first-batch slowness
        try:
            warmup_start = time.time()
            first_batch = next(iter(self.screen_loader))
            del first_batch  # Make sure to free the memory
            warmup_end = time.time()
            print(f"DATALOADER: Dataloader warmup took {warmup_end - warmup_start:.4f} seconds")
        except Exception as e:
            print(f"DATALOADER: Warmup failed: {e}")
        
        loader_end = time.time()
        print(f"DATALOADER: Loader creation took {loader_end - loader_start:.4f} seconds")
        
        total_time = time.time() - start_time
        print(f"DATALOADER: Total setup took {total_time:.4f} seconds")
        
    def run_challenge_start(self, protein_seq:str):
        # Track timing
        self.start_event.record()
        
        # Empty GPU cache
        torch.cuda.empty_cache()
        
        # Load model (this will be JIT compiled for H200)
        self.load_model()
        
        # Initialize protein data in a dedicated stream
        self.protein_dict = self.initialize_protein(protein_seq)
        
        # Record completion time
        self.end_event.record()
        torch.cuda.synchronize()
        elapsed_time = self.start_event.elapsed_time(self.end_event) / 1000  # Convert to seconds
        print(f"Model initialization completed in {elapsed_time:.2f} seconds")
        
    def run_validation(self, smiles_list:list) -> pd.DataFrame:
        """Run PSICHIC validation with detailed timing logs"""
        # Record start time
        total_start = self._log_time("validation_start")
        self.start_event.record()
        
        # Time SMILES initialization
        smiles_start = self._log_time("smiles_init_start")
        # Initialize SMILES data
        self.smiles_dict = self.initialize_smiles(smiles_list)
        smiles_end = self._log_time("smiles_initialization", smiles_start)
        
        # Time cache clearing
        cache_start = self._log_time("cache_clear_start")
        # Clear cache for inference
        torch.cuda.empty_cache()
        cache_end = self._log_time("cache_clearing", cache_start)
        
        # Time dataloader creation
        loader_start = self._log_time("loader_creation_start")
        # Create data loader with optimized batching
        self.create_screen_loader(self.protein_dict, self.smiles_dict)
        loader_end = self._log_time("loader_creation", loader_start)
        
        # Process with parallel streams and mixed precision
        stream_results = []
        
        # Time the actual model inference
        inference_start = self._log_time("inference_start")
        for i, stream in enumerate(self.streams):
            with torch.cuda.stream(stream):
                # Mark the start of stream processing
                torch.cuda.nvtx.range_push(f"stream_{i}")
                
                # Use mixed precision for faster computation
                with torch.cuda.amp.autocast(enabled=self.runtime_config.USE_MIXED_PRECISION, dtype=torch.float16):
                    # Process subset of loader
                    subset_start = self._log_time(f"subset_{i}_start")
                    subset_loader = [batch for j, batch in enumerate(self.screen_loader) if j % len(self.streams) == i]
                    
                    if subset_loader:
                        # Time the virtual screening call
                        screening_start = self._log_time(f"screening_{i}_start")
                        result_df = virtual_screening(
                            self.screen_df.copy(), 
                            self.model, 
                            subset_loader,
                            os.getcwd(),
                            save_interpret=False,
                            ligand_dict=self.smiles_dict, 
                            device=self.device,
                            save_cluster=False,
                        )
                        screening_end = self._log_time(f"screening_{i}", screening_start)
                        stream_results.append(result_df)
                    
                    subset_end = self._log_time(f"subset_{i}", subset_start)
                
                # Mark the end of stream processing
                torch.cuda.nvtx.range_pop()
        
        # Wait for all streams to complete
        sync_start = self._log_time("sync_start")
        for stream in self.streams:
            torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        sync_end = self._log_time("stream_synchronization", sync_start)
        inference_end = self._log_time("inference", inference_start)
        
        # Time results merging
        merge_start = self._log_time("merge_start")
        # Merge results
        if stream_results:
            # Find first non-empty DataFrame
            result_df = next((df for df in stream_results if not df.empty), pd.DataFrame())
            
            # Merge remaining DataFrames
            for df in stream_results:
                if not df.empty and df is not result_df:
                    # Use predicted values from each stream
                    for idx, row in df.iterrows():
                        protein = row['Protein']
                        ligand = row['Ligand']
                        match_idx = result_df[(result_df['Protein'] == protein) & (result_df['Ligand'] == ligand)].index
                        if len(match_idx) > 0:
                            if 'predicted_binding_affinity' in row and not pd.isna(row['predicted_binding_affinity']):
                                result_df.loc[match_idx, 'predicted_binding_affinity'] = row['predicted_binding_affinity']
        else:
            result_df = self.screen_df
        merge_end = self._log_time("results_merging", merge_start)
        
        # Record completion time
        total_end = self._log_time("total_validation", total_start)
        self.end_event.record()
        torch.cuda.synchronize()
        
        # Calculate elapsed time and molecules/second
        elapsed_time = self.start_event.elapsed_time(self.end_event) / 1000  # Convert to seconds
        molecules_per_second = len(smiles_list) / elapsed_time if elapsed_time > 0 else 0
        
        # Log detailed breakdown
        print(f"WRAPPER TIMING SUMMARY: Processed {len(smiles_list)} molecules in {elapsed_time:.2f} seconds ({molecules_per_second:.2f} molecules/sec)")
        print(f"WRAPPER TIMING BREAKDOWN: SMILES: {smiles_end-smiles_start:.4f}s, " + 
            f"Loader: {loader_end-loader_start:.4f}s, " +
            f"Inference: {inference_end-inference_start:.4f}s, " +
            f"Merge: {merge_end-merge_start:.4f}s")
        
        return result_df